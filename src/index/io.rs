//! Index I/O functions

use crate::error::{Error, Result};
use crate::faiss_try;
use crate::index::{CpuIndex, FromInnerPtr, IndexImpl, NativeIndex};
use faiss_sys::*;
use std::ffi::CString;
use std::ops::Deref;
use std::os::raw::c_int;
use std::ptr;
use std::sync::{Arc, Mutex};
use tokio::io::{self, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc::channel;

pub use super::io_flags::IoFlags;
use super::TryClone;

/// Write an index to a file.
///
/// # Error
///
/// This function returns an error if the description contains any byte with the value `\0` (since
/// it cannot be converted to a C string), or if the internal index writing operation fails.
pub fn write_index<I, P>(index: &I, file_name: P) -> Result<()>
where
    I: NativeIndex,
    I: CpuIndex,
    P: AsRef<str>,
{
    unsafe {
        let f = file_name.as_ref();
        let f = CString::new(f).map_err(|_| Error::BadFilePath)?;

        faiss_try(faiss_write_index_fname(index.inner_ptr(), f.as_ptr()))?;
        Ok(())
    }
}

/// Read an index from a file.
///
/// # Error
///
/// This function returns an error if the description contains any byte with the value `\0` (since
/// it cannot be converted to a C string), or if the internal index reading operation fails.
pub fn read_index<P>(file_name: P) -> Result<IndexImpl>
where
    P: AsRef<str>,
{
    unsafe {
        let f = file_name.as_ref();
        let f = CString::new(f).map_err(|_| Error::BadFilePath)?;
        let mut inner = ptr::null_mut();
        faiss_try(faiss_read_index_fname(
            f.as_ptr(),
            IoFlags::MEM_RESIDENT.into(),
            &mut inner,
        ))?;
        Ok(IndexImpl::from_inner_ptr(inner))
    }
}

/// Read an index from a file with I/O flags.
///
/// You can memory map some index types with this.
///
/// # Error
///
/// This function returns an error if the description contains any byte with the value `\0` (since
/// it cannot be converted to a C string), or if the internal index reading operation fails.
pub fn read_index_with_flags<P>(file_name: P, io_flags: IoFlags) -> Result<IndexImpl>
where
    P: AsRef<str>,
{
    unsafe {
        let f = file_name.as_ref();
        let f = CString::new(f).map_err(|_| Error::BadFilePath)?;
        let mut inner = ptr::null_mut();
        faiss_try(faiss_read_index_fname(
            f.as_ptr(),
            io_flags.0 as c_int,
            &mut inner,
        ))?;
        Ok(IndexImpl::from_inner_ptr(inner))
    }
}

pub use faiss_sys::{new_bufreceiver, new_bufsender, BufReceiver, BufSender};

pub async fn read_index_async<R>(mut r: R) -> Result<IndexImpl>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let (tx, rx) = channel(1);
    let handle = tokio::spawn(async move {
        let mut buf = [0u8; 128 * 1024];
        loop {
            let n = r.read(&mut buf).await.unwrap();
            if n == 0 {
                return;
            }
            tx.send(buf[..n].to_vec().into()).await.unwrap();
        }
    });
    let handle2 = tokio::task::spawn_blocking(move || {
        let mut br = new_bufreceiver(rx);
        let res = read_index_br(&mut br).unwrap();
        res
    });
    let index = handle2.await.unwrap();
    handle.await.unwrap();
    Ok(index)
}

/// Read an index from a BufReceiver. This is a blocking operation.
pub fn read_index_br(br: &mut BufReceiver) -> Result<IndexImpl> {
    unsafe {
        let mut inner = ptr::null_mut();
        faiss_try(faiss_read_index_br(
            br,
            IoFlags::MEM_RESIDENT.into(),
            &mut inner,
        ))?;
        Ok(IndexImpl::from_inner_ptr(inner))
    }
}

/// Write an index to a BufSender. This is a blocking operation.
pub fn write_index_bs<I>(index: &I, bs: &mut BufSender, buf_size: i32) -> Result<()>
where
    I: NativeIndex + CpuIndex,
{
    unsafe {
        faiss_try(faiss_write_index_bs(index.inner_ptr(), bs, buf_size))?;
        Ok(())
    }
}

// pub async fn write_index_async<I, W>(index: &I, mut w: W) -> Result<()>
// where
//     I: NativeIndex + Sync,
//     I: CpuIndex,
//     W: AsyncWrite + Unpin + Send + 'static,
// {
//     let (tx, mut rx) = channel(1);
//     let mut bs = new_bufsender(tx);
//     let handle = tokio::spawn(async move {
//         loop {
//             let bufitem = rx.recv().await.unwrap();
//             match bufitem {
//                 None => return,
//                 Some(buf) => w.write_all(buf.as_ref()).await.unwrap(),
//             }
//         }
//     });
//     let ptr = Mutex::new(index.inner_ptr());
//     let h2 = tokio::task::spawn_blocking(|| {
//         let ptr2 = ptr.lock().unwrap().deref();
//         write_index_bs(*ptr2, &mut bs, 128 * 1024).unwrap();
//     });
//     handle.await.unwrap();
//     Ok(())
// }

pub async fn write_index_async<I, W>(index: &I, mut w: W) -> Result<()>
where
    I: NativeIndex + CpuIndex + Sync + TryClone + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (tx, mut rx) = channel(1);
    let mut bs = new_bufsender(tx);
    let handle = tokio::spawn(async move {
        loop {
            let bufitem = rx.recv().await.unwrap();
            match bufitem {
                None => return,
                Some(buf) => w.write_all(buf.as_ref()).await.unwrap(),
            }
        }
    });

    let cloned_index = index.try_clone().unwrap();
    let handle2 = tokio::task::spawn_blocking(move || {
        let i2 = cloned_index;
        write_index_bs(&i2, &mut bs, 128 * 1024).unwrap();
    });
    handle.await.unwrap();
    handle2.await.unwrap();
    Ok(())
}

pub async fn write_index_indeximpl_async<W>(index: Arc<Mutex<IndexImpl>>, mut w: W) -> Result<()>
where
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (tx, mut rx) = channel(1);
    let mut bs = new_bufsender(tx);
    let handle = tokio::spawn(async move {
        loop {
            let bufitem = rx.recv().await.unwrap();
            match bufitem {
                None => return,
                Some(buf) => w.write_all(buf.as_ref()).await.unwrap(),
            }
        }
    });

    let handle2 = tokio::task::spawn_blocking(move || {
        let i2 = index.lock().unwrap();
        write_index_bs(&*i2, &mut bs, 128 * 1024).unwrap();
    });
    handle.await.unwrap();
    handle2.await.unwrap();
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Write};

    use bytes::{Bytes, BytesMut};
    use tokio::runtime::Runtime;
    use tokio::sync::mpsc::{channel, Receiver, Sender};

    use super::*;
    use crate::index::flat::FlatIndex;
    use crate::index::Index;
    use crate::index_factory;
    const D: u32 = 8;

    #[test]
    fn write_read() {
        let mut index = FlatIndex::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let filepath = ::std::env::temp_dir().join("test_write_read.index");
        let filename = filepath.to_str().unwrap();
        write_index(&index, filename).unwrap();
        let index = read_index(&filename).unwrap();
        assert_eq!(index.ntotal(), 5);
        ::std::fs::remove_file(&filepath).unwrap();
    }

    #[test]
    fn test_read_with_flags() {
        let index = read_index_with_flags("file_name", IoFlags::MEM_MAP | IoFlags::READ_ONLY);
        // we just want to ensure the method signature is right here
        assert!(index.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn write_read_async() {
        let mut index = FlatIndex::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let filepath = ::std::env::temp_dir().join("test_write_read_async.index");
        let filename = filepath.to_str().unwrap();
        // write_index(&index, filename).unwrap();
        // handle.await.unwrap();
        let f = tokio::fs::File::create(filename).await.unwrap();
        let res = write_index_async(&index, f).await;
        println!("res: {:?}", res);
        let f = tokio::fs::File::open(filename).await.unwrap();
        let index = read_index_async(f).await.unwrap();
        println!("done reading!");
        assert_eq!(index.ntotal(), 5);
        println!("done testing!");
        ::std::fs::remove_file(&filepath).unwrap();
        println!("done cleanup!");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn write_read_indeximpl_async() {
        let mut index = index_factory(D, "HNSW8", crate::MetricType::L2).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let filepath = ::std::env::temp_dir().join("test_write_read_async_2.index");
        let filename = filepath.to_str().unwrap();
        println!("filename: {}", filename);
        let f = tokio::fs::File::create(filename).await.unwrap();

        // wrap index in arc and mutex
        let index = Arc::new(Mutex::new(index));
        write_index_indeximpl_async(index, f).await.unwrap();
        let f = tokio::fs::File::open(filename).await.unwrap();
        let index = read_index_async(f).await.unwrap();
        assert_eq!(index.ntotal(), 5);
        ::std::fs::remove_file(&filepath).unwrap();
    }

    struct FileBufRecvr {
        f: fs::File,
        chunk_size: usize,
        tx: Sender<Bytes>,
    }

    impl FileBufRecvr {
        fn new(name: &str, chunk_size: usize) -> (Self, Receiver<Bytes>) {
            let f = fs::File::open(name).unwrap();
            let (tx, rx) = channel(1);
            (FileBufRecvr { f, chunk_size, tx }, rx)
        }

        async fn read_task(&mut self) {
            loop {
                let mut buf = BytesMut::zeroed(self.chunk_size);
                let v = self.f.read(buf.as_mut()).unwrap();
                let _ = buf.split_off(v);
                let send_buf = buf.freeze();
                if send_buf.len() == 0 {
                    return;
                }
                self.tx.send(send_buf).await.unwrap();
            }
        }
    }

    #[test]
    fn write_read_bufrecv() {
        let mut index = FlatIndex::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let filepath = ::std::env::temp_dir().join("test_write_read_bufrecv.index");
        let filename = filepath.to_str().unwrap();
        write_index(&index, filename).unwrap();

        let (mut fbr, recv) = FileBufRecvr::new(filename, 32);
        let rt = Runtime::new().unwrap();
        let handle = rt.spawn(async move { fbr.read_task().await });
        let mut br = new_bufreceiver(recv);
        let index = read_index_br(&mut br).unwrap();
        rt.block_on(handle).unwrap();
        assert_eq!(index.ntotal(), 5);
        ::std::fs::remove_file(&filepath).unwrap();
    }

    struct FileBufSndr {
        f: fs::File,
        rx: Receiver<Option<Bytes>>,
    }

    impl FileBufSndr {
        fn new(name: &str) -> (Self, Sender<Option<Bytes>>) {
            let f = fs::File::create(name).unwrap();
            let (tx, rx) = channel(1);
            (FileBufSndr { f, rx }, tx)
        }

        async fn write_task(&mut self) {
            loop {
                let bufitem = self.rx.recv().await.unwrap();
                match bufitem {
                    Some(buf) => {
                        let v = self.f.write(buf.as_ref()).unwrap();
                        if v != buf.len() {
                            panic!("wrote only {} but had {}", v, buf.len());
                        }
                    }
                    None => return,
                }
            }
        }
    }

    #[test]
    fn write_read_bufsend() {
        let mut index = FlatIndex::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let filepath = ::std::env::temp_dir().join("test_write_read_bufsend.index");
        let filename = filepath.to_str().unwrap();

        let rt = Runtime::new().unwrap();
        let (mut fbs, send) = FileBufSndr::new(filename);
        let write_handle = rt.spawn(async move { fbs.write_task().await });

        // Write the index to the bufsender
        let mut bs = new_bufsender(send);
        write_index_bs(&index, &mut bs, 32).unwrap();
        rt.block_on(write_handle).unwrap();

        // Read the index back from a bufreceiver and check.
        let (mut fbr, recv) = FileBufRecvr::new(filename, 32);
        let read_handle = rt.spawn(async move { fbr.read_task().await });
        let mut br = new_bufreceiver(recv);
        let index = read_index_br(&mut br).unwrap();
        rt.block_on(read_handle).unwrap();
        assert_eq!(index.ntotal(), 5);
        ::std::fs::remove_file(&filepath).unwrap();
    }
}

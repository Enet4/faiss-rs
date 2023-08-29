use crate::FaissIndex;

use bytes::Bytes;
use cxx::{type_id, ExternType};
use tokio::sync::mpsc::{Receiver, Sender};

unsafe impl ExternType for FaissIndex {
    type Id = type_id!("FaissIndex");
    type Kind = cxx::kind::Opaque;
}

#[cxx::bridge()]
pub mod ffi {
    extern "Rust" {
        type BufReceiver;
        fn recv_chunk(buf: &mut BufReceiver) -> &[u8];

        type BufSender;
        fn send_chunk(buf: &mut BufSender, chunk: &[u8]) -> bool;
    }

    unsafe extern "C++" {
        include!("faiss-sys/src/cpp/iobridge.h");

        type FaissIndex = crate::FaissIndex;
        unsafe fn faiss_read_index_br(
            br: &mut BufReceiver,
            io_flags: i32,
            p_out: *mut *mut FaissIndex,
        ) -> i32;

        unsafe fn faiss_write_index_bs(idx: *const FaissIndex, bs: &mut BufSender, bsz: i32)
            -> i32;
    }
}

/// BufReceiver is used to deserialize a series of Bytes into an index using
/// `faiss_read_index_br`.
pub struct BufReceiver {
    recv: Receiver<Bytes>,

    curr_chunk: Bytes,
}

pub fn new_bufreceiver(recv: Receiver<Bytes>) -> BufReceiver {
    BufReceiver {
        recv,
        curr_chunk: Bytes::new(),
    }
}

pub fn recv_chunk(buf: &mut BufReceiver) -> &[u8] {
    let next = buf.recv.blocking_recv();
    match next {
        Some(v) => {
            buf.curr_chunk = v;
            &buf.curr_chunk[..]
        }
        None => {
            buf.recv.close();
            &[]
        }
    }
}

/// BufSender is used to serialize an index into a series of Bytes using
/// `faiss_write_index_bs`.
pub struct BufSender {
    send: Sender<Option<Bytes>>,
}

pub fn new_bufsender(send: Sender<Option<Bytes>>) -> BufSender {
    BufSender { send }
}

// An empty chunk here indicates to the recv end that write has finished.
pub fn send_chunk(buf: &mut BufSender, chunk: &[u8]) -> bool {
    let b = Bytes::copy_from_slice(chunk);
    let v = {
        if b.len() == 0 {
            None
        } else {
            Some(b)
        }
    };
    match buf.send.blocking_send(v) {
        Ok(_) => true,
        Err(_e) => false,
    }
}

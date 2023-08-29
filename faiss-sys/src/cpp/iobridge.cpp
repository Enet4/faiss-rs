#include "faiss-sys/src/cpp/iobridge.h"
#include "faiss-sys/src/iobridge.rs.h"
#include <c_api/index_io_c.h>
#include <cassert>
#include <cstring>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_io.h>
#include <iostream>
#include <memory>

using faiss::Index;
using namespace std;

int faiss_read_index_br(BufReceiver &br, int io_flags, FaissIndex **p_out) {
  try {
    BufRecvReader reader(br);
    auto out = faiss::read_index(&reader, io_flags);
    *p_out = reinterpret_cast<FaissIndex *>(out);
  } catch (faiss::FaissException &e) {
    std::cerr << e.what() << '\n';
    // faiss_last_exception = std::make_exception_ptr(e);
    return -2;
  } catch (std::exception &e) {
    std::cerr << e.what() << '\n';
    // faiss_last_exception = std::make_exception_ptr(e);
    return -4;
  } catch (...) {
    std::cerr << "Unrecognized exception!\n";
    // faiss_last_exception =
    //     std::make_exception_ptr(std::runtime_error("Unknown error"));
    return -1;
  }
  return 0;
}

BufRecvReader::BufRecvReader(BufReceiver &br)
    : receiver(br), chunk_size(0), offset(0) {}

BufRecvReader::~BufRecvReader() { // close(multiBuf);
}

int BufRecvReader::fileno() {
  FAISS_THROW_MSG("BufRecvReader does not support memory mapping");
}

size_t BufRecvReader::operator()(void *ptr, size_t unitsize, size_t nitems) {
  size_t size = unitsize * nitems;
  if (size == 0)
    return 0;
  char *dst = (char *)ptr;
  size_t nb;

  { // first copy available bytes
    nb = std::min(chunk_size - offset, size);
    if (nb > 0) {
      memcpy(dst, chunk + offset, nb);
      offset += nb;
      dst += nb;
      size -= nb;
    }
  }

  // while we would like to have more data
  while (size > 0) {
    assert(offset == chunk_size); // buffer empty on input
    // try to read from main reader
    auto next_chunk_result = recv_chunk(receiver);
    chunk_size = next_chunk_result.size();
    if (chunk_size == 0) {
      break;
    }
    chunk = reinterpret_cast<const char *>(next_chunk_result.data());

    offset = 0;
    // copy remaining bytes
    size_t nb2 = std::min(chunk_size, size);
    memcpy(dst, chunk, nb2);
    offset = nb2;
    nb += nb2;
    dst += nb2;
    size -= nb2;
  }
  return nb / unitsize;
}

int faiss_write_index_bs(const FaissIndex *idx, BufSender &bs, int bsz) {
  try {
    BufSendWriter writer(bs, bsz);
    faiss::write_index(reinterpret_cast<const Index *>(idx), &writer);
  } catch (faiss::FaissException &e) {
    std::cerr << e.what() << '\n';
    // faiss_last_exception = std::make_exception_ptr(e);
    return -2;
  } catch (std::exception &e) {
    std::cerr << e.what() << '\n';
    // faiss_last_exception = std::make_exception_ptr(e);
    return -4;
  } catch (...) {
    std::cerr << "Unrecognized exception!\n";
    // faiss_last_exception =
    //     std::make_exception_ptr(std::runtime_error("Unknown error"));
    return -1;
  }
  return 0;
}

BufSendWriter::BufSendWriter(BufSender &bs, size_t bsz)
    : sender(bs), bsz(bsz), offset(0), buffer(bsz) {}

BufSendWriter::~BufSendWriter() {
  buffer.resize(offset);
  rust::Slice<const uint8_t> slice{buffer.data(), buffer.size()};
  auto result = send_chunk(sender, slice);
  FAISS_THROW_IF_NOT(result);

  // send an empty buffer to indicate write has finished
  rust::Slice<const uint8_t> slice_empty;
  auto result2 = send_chunk(sender, slice_empty);
  FAISS_THROW_IF_NOT(result2);
}

int BufSendWriter::fileno() {
  FAISS_THROW_MSG("BufSendWriter does not support memory mapping");
}

size_t BufSendWriter::operator()(const void *ptr, size_t unitsize,
                                 size_t nitems) {
  size_t size = unitsize * nitems;
  if (size == 0)
    return 0;

  const char *src = (const char *)ptr;
  size_t nb;

  { // copy as many bytes as possible to buffer
    nb = std::min(bsz - offset, size);
    memcpy(buffer.data() + offset, src, nb);
    offset += nb;
    src += nb;
    size -= nb;
  }

  while (size > 0) {
    assert(offset == bsz);
    // now we need to flush to add more bytes
    rust::Slice<const uint8_t> slice{buffer.data(), buffer.size()};
    auto result = send_chunk(sender, slice);
    FAISS_THROW_IF_NOT(result);

    size_t nb1 = std::min(bsz, size);
    memcpy(buffer.data(), src, nb1);
    offset = nb1;
    nb += nb1;
    src += nb1;
    size -= nb1;
  }
  return nb / unitsize;
}

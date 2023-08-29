#ifndef IOBRIDGE_H_
#define IOBRIDGE_H_

#pragma once
#include "rust/cxx.h"
#include <c_api/index_io_c.h>
#include <faiss/impl/io.h>
#include <memory>

using namespace faiss;

/* Defined in Rust code */
struct BufReceiver;

// std::vector<char> next_chunk(MultiBuf *mb);

// void close(MultiBuf *mb);

/* Read an index using bytes from MultiBuf - called from Rust */
int faiss_read_index_br(BufReceiver &br, int io_flags, FaissIndex **p_out);

/* BufRecvReader provides a way to read an index from a byte-stream provided
 * from rust code via the `BufReceiver` type. */
struct BufRecvReader : IOReader {
  BufReceiver &receiver;

  size_t chunk_size, offset;
  const char *chunk = nullptr;

  BufRecvReader(BufReceiver &br);
  ~BufRecvReader() override;

  size_t operator()(void *ptr, size_t size, size_t nitems) override;
  int fileno() override;
};

struct BufSender;

int faiss_write_index_bs(const FaissIndex *idx, BufSender &bs, int bsz);

struct BufSendWriter : IOWriter {
  BufSender &sender;

  size_t bsz, offset;
  std::vector<unsigned char> buffer;

  BufSendWriter(BufSender &bs, size_t bsz);
  ~BufSendWriter() override;

  size_t operator()(const void *ptr, size_t size, size_t nitems) override;
  int fileno() override;
};

#endif // IOBRIDGE_H_

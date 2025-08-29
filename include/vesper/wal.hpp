#pragma once

/** \file wal.hpp
 *  \brief Umbrella header for WAL APIs.
 *
 *  This header includes the public Write-Ahead Log (WAL) interfaces:
 *   - Frame encode/decode (frame.hpp)
 *   - Writer and recovery scanning (io.hpp)
 *   - Directory replay (replay.hpp)
 *   - Manifest helpers (manifest.hpp)
 *   - Snapshot helpers (snapshot.hpp)
 *
 *  Doxygen groups:
 *   - \defgroup wal_api WAL API
 *   - \brief Write-Ahead Log interfaces for persistence and recovery
 *   - \{
 */

#include "vesper/wal/frame.hpp"
#include "vesper/wal/io.hpp"
#include "vesper/wal/replay.hpp"
#include "vesper/wal/manifest.hpp"
#include "vesper/wal/snapshot.hpp"

/** \} */


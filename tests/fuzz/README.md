# Fuzz Targets

Placeholders for libFuzzer targets (to be added as code arrives):
- fuzz_wal_parser.cpp — feeds random frames into WAL parser, expects graceful handling
- fuzz_manifest_loader.cpp — randomized JSON inputs validated against schemas

Build integration will be added once parsers exist.


#!/bin/bash

DBS_DIR="./dbs"
CONTAINER="./envs/neo4j_v5.sif"
OUTPUT_DIR="./db_tarballs"

mkdir -p ${OUTPUT_DIR}

for DUMP_FILE in ${DBS_DIR}/*.dump; do
  
  DB_NAME=$(basename "${DUMP_FILE}" .dump)
  BUILD_DIR="./${DB_NAME}_build"

  echo "========================================"
  echo "Processing database: ${DB_NAME}"
  echo "========================================"

  mkdir -p "${BUILD_DIR}/data"

  apptainer exec \
    --bind "${DUMP_FILE}":/import/neo4j.dump \
    --bind "${BUILD_DIR}/data":/var/lib/neo4j/data \
    "${CONTAINER}" \
    neo4j-admin database load neo4j --from-path=/import --overwrite-destination=true

  apptainer exec \
    --bind "${BUILD_DIR}/data":/var/lib/neo4j/data \
    "${CONTAINER}" \
    neo4j-admin database migrate neo4j --force-btree-indexes-to-range

  echo "Compressing ${DB_NAME}.tar.gz..."
  tar -czf "${OUTPUT_DIR}/${DB_NAME}.tar.gz" -C "${BUILD_DIR}" data/

  rm -rf "${BUILD_DIR}"

  echo "Successfully built ${DB_NAME}.tar.gz"
  echo ""
done

echo "All databases have been processed and saved to ${OUTPUT_DIR}/"

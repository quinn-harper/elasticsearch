/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.RandomAccessInput;

import java.io.IOException;

/**
 * Where a field's quantized rescore copy lives in the quantized rescore file and how wide its codes are.
 * <p>
 * The copy holds one record per posting list entry, laid out in posting list order: cluster by cluster, each cluster's
 * entries sorted by document id, the same order the posting list writers use. A vector assigned to several clusters has a
 * record in each of them. After the records comes the ordinal table, one int per vector ordinal holding the index of the
 * record written for the vector's primary assignment, which is how a candidate known by ordinal reaches its record. A
 * record is the vector's codes of {@link #bits} bits each, packed as {@link QuantizedRescoreCodes} describes, followed by
 * the lower interval, upper interval and additional correction as floats and the component sum as an int. The reader
 * scores the codes against the float query and does not use the component sum. The vectors are quantized against the
 * origin rather than a centroid, so a record depends on nothing else the segment stores.
 *
 * @param bits        bits per dimension of the codes, one of {@link #SUPPORTED_BITS}
 * @param offset      where the field's records start in the file
 * @param recordCount how many records the field has, one per posting list entry
 * @param vectorCount how many vectors the field has, which is the size of the ordinal table that follows the records
 */
public record QuantizedRescoreMeta(int bits, long offset, long recordCount, int vectorCount) {

    /** The widths the copy can be written and scored at, as text for messages. */
    public static final String SUPPORTED_BITS = "2, 4 or 8";

    /** The width used when a copy is requested without one. */
    public static final int DEFAULT_BITS = 8;

    public QuantizedRescoreMeta {
        checkBits(bits);
    }

    /** Whether the copy can be written and scored at {@code bits} bits per dimension. */
    public static boolean isSupportedBits(int bits) {
        return QuantizedRescoreCodes.isSupportedWidth(bits);
    }

    /** Returns {@code bits} if the copy can be written and scored at that width, otherwise throws. */
    public static int checkBits(int bits) {
        if (isSupportedBits(bits) == false) {
            throw new IllegalArgumentException("quantized rescore bits must be one of [" + SUPPORTED_BITS + "], got [" + bits + "]");
        }
        return bits;
    }

    /** Writes {@code meta} to the field meta, or a single zero byte when the field has no copy. */
    public static void write(IndexOutput metaOutput, QuantizedRescoreMeta meta) throws IOException {
        if (meta == null) {
            metaOutput.writeByte((byte) 0);
            return;
        }
        metaOutput.writeByte((byte) meta.bits);
        metaOutput.writeLong(meta.offset);
        metaOutput.writeVLong(meta.recordCount);
        metaOutput.writeVInt(meta.vectorCount);
    }

    /** Reads what {@link #write} wrote, returning {@code null} for a field without a copy. */
    public static QuantizedRescoreMeta read(IndexInput input) throws IOException {
        final byte bits = input.readByte();
        if (bits == 0) {
            return null;
        }
        if (QuantizedRescoreCodes.isSupportedWidth(bits) == false) {
            throw new IOException("unsupported quantized rescore width [" + bits + "], expected one of [" + SUPPORTED_BITS + "]");
        }
        return new QuantizedRescoreMeta(bits, input.readLong(), input.readVLong(), input.readVInt());
    }

    /** The size of one record for vectors of {@code dimension} dimensions at {@code bits} bits per dimension. */
    public static long recordSize(int dimension, int bits) {
        return QuantizedRescoreCodes.packedLength(dimension, bits) + 3L * Float.BYTES + Integer.BYTES;
    }

    /** The size of one of this field's records for vectors of {@code dimension} dimensions. */
    public long recordSize(int dimension) {
        return recordSize(dimension, bits);
    }

    /** The total size of the field's records. */
    public long recordsLength(int dimension) {
        return recordCount * recordSize(dimension);
    }

    /** The size of the ordinal table. */
    public long tableLength() {
        return (long) vectorCount * Integer.BYTES;
    }

    /** The total size of the field's copy, records and ordinal table together. */
    public long length(int dimension) {
        return recordsLength(dimension) + tableLength();
    }

    /** The field's records as a slice of the quantized rescore file. */
    public IndexInput records(IndexInput quantizedRescoreFile, int dimension) throws IOException {
        return quantizedRescoreFile.slice("quantizedRescoreRecords", offset, recordsLength(dimension));
    }

    /** The field's ordinal table, read as {@code readInt(ord * Integer.BYTES)}. */
    public RandomAccessInput ordinalTable(IndexInput quantizedRescoreFile, int dimension) throws IOException {
        return quantizedRescoreFile.randomAccessSlice(offset + recordsLength(dimension), tableLength());
    }
}

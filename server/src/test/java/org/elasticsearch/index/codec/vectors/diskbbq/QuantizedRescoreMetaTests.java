/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;

import static org.hamcrest.Matchers.containsString;

/**
 * The field meta entry of a quantized rescore copy must round trip at every supported width, mark a field without a copy,
 * size the records and the ordinal table consistently with the width, and reject a width the reader cannot score.
 */
public class QuantizedRescoreMetaTests extends ESTestCase {

    public void testRoundTrip() throws IOException {
        QuantizedRescoreMeta meta = new QuantizedRescoreMeta(
            randomFrom(2, 4, 8),
            randomNonNegativeLong(),
            randomLongBetween(0, 1L << 40),
            randomIntBetween(0, Integer.MAX_VALUE)
        );
        assertEquals(meta, writeAndRead(meta));
        assertNull(writeAndRead(null));
    }

    public void testSizes() {
        int dimension = randomIntBetween(1, 4096);
        int bits = randomFrom(2, 4, 8);
        QuantizedRescoreMeta meta = new QuantizedRescoreMeta(bits, 0, randomLongBetween(0, 1_000_000), randomIntBetween(0, 1_000_000));
        long recordSize = (dimension * bits + Byte.SIZE - 1) / Byte.SIZE + 3L * Float.BYTES + Integer.BYTES;
        assertEquals(recordSize, QuantizedRescoreMeta.recordSize(dimension, bits));
        assertEquals(recordSize, meta.recordSize(dimension));
        assertEquals(meta.recordCount() * recordSize, meta.recordsLength(dimension));
        assertEquals((long) meta.vectorCount() * Integer.BYTES, meta.tableLength());
        assertEquals(meta.recordsLength(dimension) + meta.tableLength(), meta.length(dimension));
    }

    public void testRejectsOtherWidths() throws IOException {
        for (int width : new int[] { 1, 3, 7, 16 }) {
            try (Directory directory = new ByteBuffersDirectory()) {
                try (IndexOutput out = directory.createOutput("meta", IOContext.DEFAULT)) {
                    out.writeByte((byte) width);
                    out.writeLong(0);
                    out.writeVLong(1);
                    out.writeVInt(1);
                }
                try (IndexInput in = directory.openInput("meta", IOContext.DEFAULT)) {
                    IOException e = expectThrows(IOException.class, () -> QuantizedRescoreMeta.read(in));
                    assertThat(e.getMessage(), containsString("unsupported quantized rescore width [" + width + "]"));
                }
            }
            IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> new QuantizedRescoreMeta(width, 0, 1, 1));
            assertThat(e.getMessage(), containsString("quantized rescore bits must be one of [2, 4 or 8], got [" + width + "]"));
        }
    }

    private static QuantizedRescoreMeta writeAndRead(QuantizedRescoreMeta meta) throws IOException {
        try (Directory directory = new ByteBuffersDirectory()) {
            try (IndexOutput out = directory.createOutput("meta", IOContext.DEFAULT)) {
                QuantizedRescoreMeta.write(out, meta);
            }
            try (IndexInput in = directory.openInput("meta", IOContext.DEFAULT)) {
                QuantizedRescoreMeta read = QuantizedRescoreMeta.read(in);
                assertEquals("the entry must be consumed completely", in.length(), in.getFilePointer());
                return read;
            }
        }
    }
}

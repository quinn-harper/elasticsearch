/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.elasticsearch.test.ESTestCase;

/**
 * The code layout of a quantized rescore record must round trip at every supported width, including dimensions that do
 * not fill the last byte, and size the packed codes as the meta expects.
 */
public class QuantizedRescoreCodesTests extends ESTestCase {

    public void testPackAndUnpackRoundTrip() {
        for (int bits : new int[] { 2, 4, 8 }) {
            int dimension = randomIntBetween(1, 1024);
            int[] codes = new int[dimension];
            for (int i = 0; i < dimension; i++) {
                codes[i] = randomIntBetween(0, (1 << bits) - 1);
            }
            byte[] packed = new byte[QuantizedRescoreCodes.packedLength(dimension, bits)];
            QuantizedRescoreCodes.pack(codes, dimension, bits, packed);
            byte[] unpacked = new byte[dimension];
            QuantizedRescoreCodes.unpack(packed, dimension, bits, unpacked);
            for (int i = 0; i < dimension; i++) {
                assertEquals("bits " + bits + " dimension " + i, codes[i], unpacked[i] + QuantizedRescoreCodes.codeOffset(bits));
            }
        }
    }

    public void testPackedLength() {
        assertEquals(1, QuantizedRescoreCodes.packedLength(1, 2));
        assertEquals(1, QuantizedRescoreCodes.packedLength(4, 2));
        assertEquals(2, QuantizedRescoreCodes.packedLength(5, 2));
        assertEquals(1, QuantizedRescoreCodes.packedLength(1, 4));
        assertEquals(1, QuantizedRescoreCodes.packedLength(2, 4));
        assertEquals(2, QuantizedRescoreCodes.packedLength(3, 4));
        assertEquals(7, QuantizedRescoreCodes.packedLength(7, 8));
        assertEquals(128, QuantizedRescoreCodes.packedLength(512, 2));
    }

    public void testWidthsAndOffsets() {
        for (int bits = 0; bits <= 9; bits++) {
            assertEquals("bits " + bits, bits == 2 || bits == 4 || bits == 8, QuantizedRescoreCodes.isSupportedWidth(bits));
        }
        assertEquals(0, QuantizedRescoreCodes.codeOffset(2));
        assertEquals(0, QuantizedRescoreCodes.codeOffset(4));
        assertEquals(128, QuantizedRescoreCodes.codeOffset(8));
    }
}

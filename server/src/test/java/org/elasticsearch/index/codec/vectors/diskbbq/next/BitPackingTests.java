/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.elasticsearch.test.ESTestCase;

import static org.hamcrest.Matchers.equalTo;

public class BitPackingTests extends ESTestCase {

    public void testRoundTripForEveryBitWidth() {
        for (int bitsPerValue = 1; bitsPerValue <= 8; bitsPerValue++) {
            final int numberOfValues = randomIntBetween(1, 300);
            final int[] values = new int[numberOfValues];
            for (int i = 0; i < numberOfValues; i++) {
                values[i] = randomIntBetween(0, (1 << bitsPerValue) - 1);
            }
            final int offset = randomIntBetween(0, 5);
            final byte[] packed = new byte[offset + BitPacking.packedLength(numberOfValues, bitsPerValue)];
            BitPacking.pack(values, numberOfValues, bitsPerValue, packed, offset);

            final int[] unpacked = new int[numberOfValues];
            BitPacking.unpack(packed, offset, numberOfValues, bitsPerValue, unpacked);
            assertArrayEquals("bitsPerValue=" + bitsPerValue, values, unpacked);
        }
    }

    public void testPackedLengthRoundsUpToWholeBytes() {
        assertThat(BitPacking.packedLength(8, 1), equalTo(1));
        assertThat(BitPacking.packedLength(9, 1), equalTo(2));
        assertThat(BitPacking.packedLength(5, 7), equalTo(5));
        assertThat(BitPacking.packedLength(3, 3), equalTo(2));
        assertThat(BitPacking.packedLength(768, 4), equalTo(384));
    }

    public void testExtremeValuesDoNotBleedIntoNeighbours() {
        for (int bitsPerValue = 1; bitsPerValue <= 8; bitsPerValue++) {
            final int maxValue = (1 << bitsPerValue) - 1;
            final int[] values = new int[] { maxValue, 0, maxValue, 0, 0, maxValue, maxValue, 0, maxValue };
            final byte[] packed = new byte[BitPacking.packedLength(values.length, bitsPerValue)];
            BitPacking.pack(values, values.length, bitsPerValue, packed, 0);
            final int[] unpacked = new int[values.length];
            BitPacking.unpack(packed, 0, values.length, bitsPerValue, unpacked);
            assertArrayEquals("bitsPerValue=" + bitsPerValue, values, unpacked);
        }
    }
}

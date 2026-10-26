/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

/**
 * Packs small unsigned integers of any width from 1 to 8 bits into a byte array, least significant bits first.
 * The quantized rescoring section needs this because its code widths (the OSQ bits and the user chosen rescore
 * bits) are not limited to the widths the posting list layouts support.
 */
final class BitPacking {

    private BitPacking() {}

    static int packedLength(int numberOfValues, int bitsPerValue) {
        return (numberOfValues * bitsPerValue + 7) / 8;
    }

    static void pack(int[] values, int numberOfValues, int bitsPerValue, byte[] destination, int destinationOffset) {
        assert bitsPerValue >= 1 && bitsPerValue <= Byte.SIZE : "bitsPerValue must be between 1 and 8, got " + bitsPerValue;
        long accumulator = 0;
        int accumulatedBits = 0;
        int position = destinationOffset;
        for (int i = 0; i < numberOfValues; i++) {
            assert values[i] >= 0 && values[i] < (1 << bitsPerValue) : "value " + values[i] + " does not fit in " + bitsPerValue + " bits";
            accumulator |= ((long) values[i]) << accumulatedBits;
            accumulatedBits += bitsPerValue;
            while (accumulatedBits >= Byte.SIZE) {
                destination[position++] = (byte) accumulator;
                accumulator >>>= Byte.SIZE;
                accumulatedBits -= Byte.SIZE;
            }
        }
        if (accumulatedBits > 0) {
            destination[position] = (byte) accumulator;
        }
    }

    static void unpack(byte[] source, int sourceOffset, int numberOfValues, int bitsPerValue, int[] destination) {
        assert bitsPerValue >= 1 && bitsPerValue <= Byte.SIZE : "bitsPerValue must be between 1 and 8, got " + bitsPerValue;
        final int mask = (1 << bitsPerValue) - 1;
        long accumulator = 0;
        int availableBits = 0;
        int position = sourceOffset;
        for (int i = 0; i < numberOfValues; i++) {
            while (availableBits < bitsPerValue) {
                accumulator |= ((long) (source[position++] & 0xFF)) << availableBits;
                availableBits += Byte.SIZE;
            }
            destination[i] = (int) (accumulator & mask);
            accumulator >>>= bitsPerValue;
            availableBits -= bitsPerValue;
        }
    }
}

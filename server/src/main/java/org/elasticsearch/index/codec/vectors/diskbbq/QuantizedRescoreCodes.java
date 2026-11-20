/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import java.util.Arrays;

/**
 * How the codes of a quantized rescore record are laid out in bytes, see {@link QuantizedRescoreMeta}. Codes of 2 and 4
 * bits are packed several to a byte, the lowest dimension in the lowest bits. Codes of 8 bits take a byte each but are
 * shifted by {@link #codeOffset} so that the full code range fits a signed byte, which lets the reader use a signed byte
 * dot product and add the offset back through the query component sum.
 */
final class QuantizedRescoreCodes {

    private QuantizedRescoreCodes() {}

    /** Whether {@code bits} is a width the copy can be written and scored at. */
    static boolean isSupportedWidth(int bits) {
        return bits == 2 || bits == 4 || bits == 8;
    }

    /** The bytes the codes of one vector of {@code dimension} dimensions take at {@code bits} bits each. */
    static int packedLength(int dimension, int bits) {
        return (dimension * bits + Byte.SIZE - 1) / Byte.SIZE;
    }

    /** What the writer subtracts from each code before storing it and the reader adds back when scoring. */
    static int codeOffset(int bits) {
        return bits == Byte.SIZE ? 1 << (Byte.SIZE - 1) : 0;
    }

    /** Stores the first {@code dimension} codes of {@code codes}, each of {@code bits} bits, into {@code packed}. */
    static void pack(int[] codes, int dimension, int bits, byte[] packed) {
        if (bits == Byte.SIZE) {
            final int offset = codeOffset(bits);
            for (int i = 0; i < dimension; i++) {
                packed[i] = (byte) (codes[i] - offset);
            }
            return;
        }
        final int codesPerByte = Byte.SIZE / bits;
        Arrays.fill(packed, 0, packedLength(dimension, bits), (byte) 0);
        for (int i = 0; i < dimension; i++) {
            packed[i / codesPerByte] |= (byte) (codes[i] << ((i % codesPerByte) * bits));
        }
    }

    /**
     * Restores one byte per code from what {@link #pack} stored. Codes of fewer than 8 bits come back as written; codes of
     * 8 bits stay shifted by {@link #codeOffset}.
     */
    static void unpack(byte[] packed, int dimension, int bits, byte[] codes) {
        if (bits == Byte.SIZE) {
            System.arraycopy(packed, 0, codes, 0, dimension);
            return;
        }
        final int codesPerByte = Byte.SIZE / bits;
        final int mask = (1 << bits) - 1;
        for (int i = 0; i < dimension; i++) {
            codes[i] = (byte) ((packed[i / codesPerByte] >>> ((i % codesPerByte) * bits)) & mask);
        }
    }
}

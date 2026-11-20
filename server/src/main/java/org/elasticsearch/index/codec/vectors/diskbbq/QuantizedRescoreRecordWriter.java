/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.IndexOutput;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;

import java.io.IOException;

/**
 * Writes the records of a field's quantized rescore copy, see {@link QuantizedRescoreMeta} for the layout. Each vector is
 * quantized on its own with the optimized scalar quantizer at the copy's width against the origin, which the reader undoes
 * at search time by rebuilding the vector from its interval and codes. The codes are stored as {@link QuantizedRescoreCodes}
 * lays them out.
 */
final class QuantizedRescoreRecordWriter {

    private final IndexOutput out;
    private final OptimizedScalarQuantizer quantizer;
    private final int bits;
    private final float[] origin;
    private final float[] centered;
    private final int[] codes;
    private final byte[] record;

    QuantizedRescoreRecordWriter(FieldInfo fieldInfo, int bits, IndexOutput out) {
        final int dimension = fieldInfo.getVectorDimension();
        this.out = out;
        this.quantizer = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
        this.bits = QuantizedRescoreMeta.checkBits(bits);
        this.origin = new float[dimension];
        this.centered = new float[dimension];
        this.codes = new int[dimension];
        this.record = new byte[QuantizedRescoreCodes.packedLength(dimension, bits)];
    }

    void write(float[] vector) throws IOException {
        final OptimizedScalarQuantizer.QuantizationResult result = quantizer.scalarQuantize(vector, centered, codes, (byte) bits, origin);
        QuantizedRescoreCodes.pack(codes, codes.length, bits, record);
        out.writeBytes(record, record.length);
        out.writeInt(Float.floatToIntBits(result.lowerInterval()));
        out.writeInt(Float.floatToIntBits(result.upperInterval()));
        out.writeInt(Float.floatToIntBits(result.additionalCorrection()));
        out.writeInt(result.quantizedComponentSum());
    }
}

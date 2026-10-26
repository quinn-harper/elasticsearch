/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.elasticsearch.Build;
import org.elasticsearch.ElasticsearchStatusException;
import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;

import java.io.IOException;
import java.util.Objects;

public class RescoreVectorBuilder implements Writeable, ToXContentObject {

    public static final ParseField OVERSAMPLE_FIELD = new ParseField("oversample");
    public static final ParseField QUANTIZED_FIELD = new ParseField("quantized");
    public static final float NO_OVERSAMPLE = 0.0F;
    public static final float MIN_OVERSAMPLE = 1.0F;
    private static final ConstructingObjectParser<RescoreVectorBuilder, Void> PARSER = new ConstructingObjectParser<>(
        "rescore_vector",
        args -> new RescoreVectorBuilder((Float) args[0], (Boolean) args[1])
    );

    static {
        PARSER.declareFloat(ConstructingObjectParser.constructorArg(), OVERSAMPLE_FIELD);
        PARSER.declareBoolean(ConstructingObjectParser.optionalConstructorArg(), QUANTIZED_FIELD);
    }

    public static final TransportVersion RESCORE_VECTOR_ALLOW_ZERO = TransportVersion.fromName("rescore_vector_allow_zero");
    public static final TransportVersion RESCORE_VECTOR_QUANTIZED = TransportVersion.fromName("rescore_vector_quantized");

    // Oversample is required as of now, quantized is an optional opt out of rescoring with stored quantized residual codes
    private final float oversample;
    private final Boolean quantized;

    public RescoreVectorBuilder(float numCandidatesFactor) {
        this(numCandidatesFactor, null);
    }

    /**
     * @param quantized {@code false} to rescore with the raw vectors even when the segment stores quantized residual codes,
     *                  {@code true} or {@code null} to use the residual codes when the segment has them
     */
    public RescoreVectorBuilder(float numCandidatesFactor, Boolean quantized) {
        Objects.requireNonNull(numCandidatesFactor, "[" + OVERSAMPLE_FIELD.getPreferredName() + "] must be set");
        if (numCandidatesFactor < MIN_OVERSAMPLE && numCandidatesFactor != NO_OVERSAMPLE) {
            throw new IllegalArgumentException("[" + OVERSAMPLE_FIELD.getPreferredName() + "] must be >= " + MIN_OVERSAMPLE + " or 0");
        }
        if (quantized != null && Build.current().isSnapshot() == false) {
            throw new IllegalArgumentException("[" + QUANTIZED_FIELD.getPreferredName() + "] is only supported on snapshot builds");
        }
        this.oversample = numCandidatesFactor;
        this.quantized = quantized;
    }

    public RescoreVectorBuilder(StreamInput in) throws IOException {
        this.oversample = in.readFloat();
        this.quantized = in.getTransportVersion().supports(RESCORE_VECTOR_QUANTIZED) ? in.readOptionalBoolean() : null;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        // We don't want to serialize a `0` oversample to a node that doesn't know what to do with it.
        if (oversample == NO_OVERSAMPLE && out.getTransportVersion().supports(RESCORE_VECTOR_ALLOW_ZERO) == false) {
            throw new ElasticsearchStatusException(
                "[rescore_vector] does not support a 0 for ["
                    + OVERSAMPLE_FIELD.getPreferredName()
                    + "] before version ["
                    + RESCORE_VECTOR_ALLOW_ZERO.toReleaseVersion()
                    + "]",
                RestStatus.BAD_REQUEST
            );
        }
        out.writeFloat(oversample);
        // A node that does not know the field only rescores with raw vectors, which is what quantized=false asks for
        if (out.getTransportVersion().supports(RESCORE_VECTOR_QUANTIZED)) {
            out.writeOptionalBoolean(quantized);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(OVERSAMPLE_FIELD.getPreferredName(), oversample);
        if (quantized != null) {
            builder.field(QUANTIZED_FIELD.getPreferredName(), quantized);
        }
        builder.endObject();
        return builder;
    }

    public static RescoreVectorBuilder fromXContent(XContentParser parser) {
        return PARSER.apply(parser, null);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RescoreVectorBuilder that = (RescoreVectorBuilder) o;
        return Objects.equals(oversample, that.oversample) && Objects.equals(quantized, that.quantized);
    }

    @Override
    public int hashCode() {
        return Objects.hash(oversample, quantized);
    }

    public float oversample() {
        return oversample;
    }

    /** {@code null} when the request did not say, which means to use quantized residual codes where a segment has them. */
    public Boolean quantized() {
        return quantized;
    }
}

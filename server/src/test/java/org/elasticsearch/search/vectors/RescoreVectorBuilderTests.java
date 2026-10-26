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
import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.Strings;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.test.TransportVersionUtils;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xcontent.json.JsonXContent;

import java.io.IOException;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.nullValue;

public class RescoreVectorBuilderTests extends ESTestCase {

    public void testParseOversampleOnlyLeavesQuantizedUnset() throws IOException {
        try (XContentParser parser = createParser(JsonXContent.jsonXContent, "{\"oversample\": 2.5}")) {
            RescoreVectorBuilder builder = RescoreVectorBuilder.fromXContent(parser);
            assertThat(builder.oversample(), equalTo(2.5f));
            assertThat(builder.quantized(), nullValue());
        }
    }

    public void testParseQuantizedOptOut() throws IOException {
        assumeTrue("quantized is only accepted on snapshot builds", Build.current().isSnapshot());
        try (XContentParser parser = createParser(JsonXContent.jsonXContent, "{\"oversample\": 3, \"quantized\": false}")) {
            RescoreVectorBuilder builder = RescoreVectorBuilder.fromXContent(parser);
            assertThat(builder.oversample(), equalTo(3f));
            assertThat(builder.quantized(), equalTo(false));
        }
    }

    public void testQuantizedIsWrittenBackToXContent() throws IOException {
        assumeTrue("quantized is only accepted on snapshot builds", Build.current().isSnapshot());
        assertThat(Strings.toString(new RescoreVectorBuilder(2f, false)), containsString("\"quantized\":false"));
        assertThat(Strings.toString(new RescoreVectorBuilder(2f)), not(containsString("quantized")));
    }

    public void testSerializationRoundTripKeepsQuantized() throws IOException {
        assumeTrue("quantized is only accepted on snapshot builds", Build.current().isSnapshot());
        for (Boolean quantized : new Boolean[] { null, true, false }) {
            RescoreVectorBuilder original = new RescoreVectorBuilder(randomFrom(0f, 1f, 4.5f), quantized);
            RescoreVectorBuilder copy = copyWriteable(
                original,
                writableRegistry(),
                RescoreVectorBuilder::new,
                TransportVersion.current()
            );
            assertThat(copy, equalTo(original));
            assertThat(copy.quantized(), equalTo(quantized));
        }
    }

    public void testNodesThatDoNotKnowQuantizedNeverSeeIt() throws IOException {
        assumeTrue("quantized is only accepted on snapshot builds", Build.current().isSnapshot());
        TransportVersion older = TransportVersionUtils.randomVersionNotSupporting(RescoreVectorBuilder.RESCORE_VECTOR_QUANTIZED);
        RescoreVectorBuilder original = new RescoreVectorBuilder(2f, false);
        RescoreVectorBuilder copy = copyWriteable(original, writableRegistry(), RescoreVectorBuilder::new, older);
        assertThat(copy.oversample(), equalTo(2f));
        assertThat(copy.quantized(), nullValue());
    }
}

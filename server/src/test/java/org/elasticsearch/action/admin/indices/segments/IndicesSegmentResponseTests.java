/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.action.admin.indices.segments;

import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.elasticsearch.cluster.routing.ShardRouting;
import org.elasticsearch.cluster.routing.ShardRoutingState;
import org.elasticsearch.cluster.routing.TestShardRouting;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.xcontent.ChunkedToXContent;
import org.elasticsearch.index.engine.Segment;
import org.elasticsearch.test.AbstractChunkedSerializingTestCase;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xcontent.XContentBuilder;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.xcontent.ToXContent.EMPTY_PARAMS;
import static org.elasticsearch.xcontent.XContentFactory.jsonBuilder;

public class IndicesSegmentResponseTests extends ESTestCase {

    public void testToXContentSerialiationWithSortedFields() throws Exception {
        ShardRouting shardRouting = TestShardRouting.newShardRouting("foo", 0, "node_id", true, ShardRoutingState.STARTED);
        Segment segment = new Segment("my");

        SortField sortField = new SortField("foo", SortField.Type.STRING);
        sortField.setMissingValue(SortField.STRING_LAST);
        segment.segmentSort = new Sort(sortField);

        ShardSegments shardSegments = new ShardSegments(shardRouting, Collections.singletonList(segment));
        IndicesSegmentResponse response = new IndicesSegmentResponse(
            new ShardSegments[] { shardSegments },
            1,
            1,
            0,
            Collections.emptyList()
        );
        try (XContentBuilder builder = jsonBuilder()) {
            ChunkedToXContent.wrapAsToXContent(response).toXContent(builder, EMPTY_PARAMS);
        }
    }

    public void testToXContentWithAutoCalibration() throws Exception {
        ShardRouting shardRouting = TestShardRouting.newShardRouting("idx", 0, "node_id", true, ShardRoutingState.STARTED);
        Segment segment = new Segment("_1");
        segment.autoCalibrationInfo = Map.of("emb", new Segment.FieldCalibrationInfo(true, (byte) 1, (byte) 4, 1.25f, true));
        ShardSegments shardSegments = new ShardSegments(shardRouting, Collections.singletonList(segment));
        IndicesSegmentResponse response = new IndicesSegmentResponse(
            new ShardSegments[] { shardSegments },
            1,
            1,
            0,
            Collections.emptyList()
        );
        String json;
        try (XContentBuilder builder = jsonBuilder()) {
            ChunkedToXContent.wrapAsToXContent(response).toXContent(builder, EMPTY_PARAMS);
            json = Strings.toString(builder);
        }
        assertThat(json, org.hamcrest.Matchers.containsString("\"auto_calibration\""));
        assertThat(json, org.hamcrest.Matchers.containsString("\"emb\""));
        assertThat(json, org.hamcrest.Matchers.containsString("\"calibrated\":true"));
        assertThat(json, org.hamcrest.Matchers.containsString("\"bits\":1"));
        assertThat(json, org.hamcrest.Matchers.containsString("\"queryBits\":4"));
        assertThat(json, org.hamcrest.Matchers.containsString("\"precondition\":true"));
    }

    public void testToXContentAutoCalibrationAbsentWhenNull() throws Exception {
        ShardRouting shardRouting = TestShardRouting.newShardRouting("idx", 0, "node_id", true, ShardRoutingState.STARTED);
        Segment segment = new Segment("_1");
        // autoCalibrationInfo is null by default
        ShardSegments shardSegments = new ShardSegments(shardRouting, Collections.singletonList(segment));
        IndicesSegmentResponse response = new IndicesSegmentResponse(
            new ShardSegments[] { shardSegments },
            1,
            1,
            0,
            Collections.emptyList()
        );
        String json;
        try (XContentBuilder builder = jsonBuilder()) {
            ChunkedToXContent.wrapAsToXContent(response).toXContent(builder, EMPTY_PARAMS);
            json = Strings.toString(builder);
        }
        assertThat(json, org.hamcrest.Matchers.not(org.hamcrest.Matchers.containsString("auto_calibration")));
    }

    public void testChunking() {
        final int indices = randomIntBetween(1, 10);
        final List<ShardRouting> routings = new ArrayList<>(indices);
        for (int i = 0; i < indices; i++) {
            routings.add(TestShardRouting.newShardRouting("index-" + i, 0, "node_id", true, ShardRoutingState.STARTED));
        }
        Segment segment = new Segment("my");
        SortField sortField = new SortField("foo", SortField.Type.STRING);
        sortField.setMissingValue(SortField.STRING_LAST);
        segment.segmentSort = new Sort(sortField);
        AbstractChunkedSerializingTestCase.assertChunkCount(
            new IndicesSegmentResponse(
                routings.stream().map(routing -> new ShardSegments(routing, List.of(segment))).toArray(ShardSegments[]::new),
                indices,
                indices,
                0,
                Collections.emptyList()
            ),
            response -> 11 * response.getIndices().size() + 4
        );
    }
}

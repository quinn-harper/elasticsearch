/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.engine;

import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.util.Version;
import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.io.stream.BytesStreamOutput;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.test.TransportVersionUtils;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;

public class SegmentTests extends ESTestCase {
    static SortField randomSortField() {
        return switch (between(0, 2)) {
            case 0 -> {
                SortedNumericSortField field = new SortedNumericSortField(
                    randomAlphaOfLengthBetween(1, 10),
                    SortField.Type.INT,
                    randomBoolean(),
                    randomBoolean() ? SortedNumericSelector.Type.MAX : SortedNumericSelector.Type.MIN
                );
                if (randomBoolean()) {
                    field.setMissingValue(randomInt());
                }
                yield field;
            }
            case 1 -> {
                SortedSetSortField field = new SortedSetSortField(
                    randomAlphaOfLengthBetween(1, 10),
                    randomBoolean(),
                    randomBoolean() ? SortedSetSelector.Type.MAX : SortedSetSelector.Type.MIN
                );
                if (randomBoolean()) {
                    field.setMissingValue(randomBoolean() ? SortedSetSortField.STRING_FIRST : SortedSetSortField.STRING_LAST);
                }
                yield field;
            }
            case 2 -> {
                SortField field = new SortField(randomAlphaOfLengthBetween(1, 10), SortField.Type.STRING, randomBoolean());
                if (randomBoolean()) {
                    field.setMissingValue(randomBoolean() ? SortedSetSortField.STRING_FIRST : SortedSetSortField.STRING_LAST);
                }
                yield field;
            }
            default -> throw new UnsupportedOperationException();
        };
    }

    static Sort randomIndexSort() {
        if (randomBoolean()) {
            return null;
        }
        int size = randomIntBetween(1, 5);
        SortField[] fields = new SortField[size];
        for (int i = 0; i < size; i++) {
            fields[i] = randomSortField();
        }
        return new Sort(fields);
    }

    static Segment.FieldCalibrationInfo randomFieldCalibrationInfo() {
        boolean calibrated = randomBoolean();
        byte bits = calibrated ? randomFrom((byte) 1, (byte) 2, (byte) 4, (byte) 7) : (byte) -1;
        byte queryBits = calibrated ? (bits == 7 ? (byte) 7 : (byte) 4) : (byte) -1;
        float oversample = calibrated ? randomFloatBetween(1.0f, 3.0f, true) : Float.NaN;
        boolean precondition = randomBoolean();
        return new Segment.FieldCalibrationInfo(calibrated, bits, queryBits, oversample, precondition);
    }

    static Segment randomSegment() {
        Segment segment = new Segment(randomAlphaOfLength(10));
        segment.committed = randomBoolean();
        segment.search = randomBoolean();
        segment.sizeInBytes = randomNonNegativeLong();
        segment.docCount = randomIntBetween(1, Integer.MAX_VALUE);
        segment.delDocCount = randomIntBetween(0, segment.docCount);
        segment.version = Version.LUCENE_9_0_0;
        segment.compound = randomBoolean();
        segment.mergeId = randomAlphaOfLengthBetween(1, 10);
        segment.segmentSort = randomIndexSort();
        if (randomBoolean()) {
            segment.attributes = Collections.singletonMap("foo", "bar");
        }
        if (randomBoolean()) {
            segment.autoCalibrationInfo = Map.of(randomAlphaOfLength(5), randomFieldCalibrationInfo());
        }
        return segment;
    }

    public void testSerialization() throws IOException {
        for (int i = 0; i < 20; i++) {
            Segment segment = randomSegment();
            BytesStreamOutput output = new BytesStreamOutput();
            segment.writeTo(output);
            output.flush();
            StreamInput input = output.bytes().streamInput();
            Segment deserialized = new Segment(input);
            assertTrue(isSegmentEquals(deserialized, segment));
        }
    }

    public void testSerializationOldTransportVersion() throws IOException {
        TransportVersion old = TransportVersionUtils.getPreviousVersion(TransportVersion.current());
        for (int i = 0; i < 10; i++) {
            Segment segment = randomSegment();
            // populate autoCalibrationInfo so we can verify it is not sent to old nodes
            segment.autoCalibrationInfo = Map.of("field", randomFieldCalibrationInfo());
            BytesStreamOutput output = new BytesStreamOutput();
            output.setTransportVersion(old);
            segment.writeTo(output);
            output.flush();
            StreamInput input = output.bytes().streamInput();
            input.setTransportVersion(old);
            Segment deserialized = new Segment(input);
            assertNull(deserialized.autoCalibrationInfo);
        }
    }

    static boolean isFieldCalibrationInfoEquals(Segment.FieldCalibrationInfo a, Segment.FieldCalibrationInfo b) {
        if (a == b) return true;
        if (a == null || b == null) return false;
        return a.calibrated == b.calibrated
            && a.bits == b.bits
            && a.queryBits == b.queryBits
            && (Float.isNaN(a.oversample) ? Float.isNaN(b.oversample) : Float.compare(a.oversample, b.oversample) == 0)
            && a.precondition == b.precondition;
    }

    static boolean isSegmentEquals(Segment seg1, Segment seg2) {
        if (seg1.autoCalibrationInfo == null && seg2.autoCalibrationInfo != null) return false;
        if (seg1.autoCalibrationInfo != null && seg2.autoCalibrationInfo == null) return false;
        if (seg1.autoCalibrationInfo != null) {
            if (seg1.autoCalibrationInfo.size() != seg2.autoCalibrationInfo.size()) return false;
            for (var entry : seg1.autoCalibrationInfo.entrySet()) {
                if (isFieldCalibrationInfoEquals(entry.getValue(), seg2.autoCalibrationInfo.get(entry.getKey())) == false) {
                    return false;
                }
            }
        }
        return seg1.docCount == seg2.docCount
            && seg1.delDocCount == seg2.delDocCount
            && seg1.committed == seg2.committed
            && seg1.search == seg2.search
            && Objects.equals(seg1.version, seg2.version)
            && Objects.equals(seg1.compound, seg2.compound)
            && seg1.sizeInBytes == seg2.sizeInBytes
            && seg1.getGeneration() == seg2.getGeneration()
            && seg1.getName().equals(seg2.getName())
            && seg1.getMergeId().equals(seg2.getMergeId())
            && Objects.equals(seg1.segmentSort, seg2.segmentSort);
    }
}

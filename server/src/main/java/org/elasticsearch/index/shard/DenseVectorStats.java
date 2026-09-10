/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.shard;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.xcontent.ToXContentFragment;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.elasticsearch.common.unit.ByteSizeValue.ofBytes;

/**
 * Statistics about indexed dense vector
 */
public class DenseVectorStats implements Writeable, ToXContentFragment {

    private static final TransportVersion DENSE_VECTOR_OFF_HEAP_STATS = TransportVersion.fromName("dense_vector_off_heap_stats");
    private static final TransportVersion DENSE_VECTOR_AUTO_CALIBRATION_STATS = TransportVersion.fromName(
        "dense_vector_auto_calibration_stats"
    );

    /**
     * Groups segments by their auto-calibration outcome for a single field.
     * When {@code calibrated=false}, the remaining fields are unused and should be ignored.
     */
    public record AutoCalibrationKey(boolean calibrated, byte bits, byte queryBits, float oversample, boolean precondition) {
        public static final AutoCalibrationKey UNCALIBRATED = new AutoCalibrationKey(false, (byte) 0, (byte) 0, Float.NaN, false);

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o instanceof AutoCalibrationKey other == false) return false;
            if (calibrated != other.calibrated) return false;
            if (calibrated == false) return true;
            return bits == other.bits
                && queryBits == other.queryBits
                && Float.floatToIntBits(oversample) == Float.floatToIntBits(other.oversample)
                && precondition == other.precondition;
        }

        @Override
        public int hashCode() {
            if (calibrated == false) return Boolean.hashCode(false);
            return Objects.hash(calibrated, bits, queryBits, Float.floatToIntBits(oversample), precondition);
        }
    }

    public record AutoCalibrationEntry(AutoCalibrationKey key, int segmentCount, long documentCount, long sizeInBytes) {
        public AutoCalibrationEntry merge(AutoCalibrationEntry other) {
            assert key.equals(other.key);
            return new AutoCalibrationEntry(key, segmentCount + other.segmentCount, documentCount + other.documentCount, sizeInBytes + other.sizeInBytes);
        }
    }

    private long valueCount = 0;

    /** Per-field off-heap desired memory byte size, categorized by file extension. */
    Map<String, Map<String, Long>> offHeapStats;

    /** Per-field auto-calibration configuration groups with aggregated segment, document, and size stats. */
    Map<String, Map<AutoCalibrationKey, AutoCalibrationEntry>> autoCalibrationStats;

    public DenseVectorStats() {}

    public DenseVectorStats(long count) {
        this(count, null, null);
    }

    public DenseVectorStats(long count, Map<String, Map<String, Long>> offHeapStats) {
        this(count, offHeapStats, null);
    }

    public DenseVectorStats(
        long count,
        Map<String, Map<String, Long>> offHeapStats,
        Map<String, Map<AutoCalibrationKey, AutoCalibrationEntry>> autoCalibrationStats
    ) {
        this.valueCount = count;
        this.offHeapStats = offHeapStats;
        this.autoCalibrationStats = autoCalibrationStats;
    }

    public DenseVectorStats(StreamInput in) throws IOException {
        this.valueCount = in.readVLong();
        if (in.getTransportVersion().supports(DENSE_VECTOR_OFF_HEAP_STATS)) {
            this.offHeapStats = readOptionalOffHeapStats(in);
        }
        if (in.getTransportVersion().supports(DENSE_VECTOR_AUTO_CALIBRATION_STATS)) {
            this.autoCalibrationStats = readOptionalAutoCalibrationStats(in);
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVLong(valueCount);
        if (out.getTransportVersion().supports(DENSE_VECTOR_OFF_HEAP_STATS)) {
            writeOptionalOffHeapStats(out);
        }
        if (out.getTransportVersion().supports(DENSE_VECTOR_AUTO_CALIBRATION_STATS)) {
            writeOptionalAutoCalibrationStats(out);
        }
    }

    private Map<String, Map<String, Long>> readOptionalOffHeapStats(StreamInput in) throws IOException {
        if (in.readBoolean()) {
            return in.readMap(v -> in.readMap(StreamInput::readLong));
        } else {
            return null;
        }
    }

    private void writeOptionalOffHeapStats(StreamOutput out) throws IOException {
        if (offHeapStats != null) {
            out.writeBoolean(true);
            out.writeMap(offHeapStats, StreamOutput::writeString, DenseVectorStats::writeFieldStatsMap);
        } else {
            out.writeBoolean(false);
        }
    }

    static void writeFieldStatsMap(StreamOutput out, Map<String, Long> map) throws IOException {
        out.writeMap(map, StreamOutput::writeString, StreamOutput::writeLong);
    }

    private Map<String, Map<AutoCalibrationKey, AutoCalibrationEntry>> readOptionalAutoCalibrationStats(StreamInput in)
        throws IOException {
        if (in.readBoolean() == false) {
            return null;
        }
        int fieldCount = in.readVInt();
        Map<String, Map<AutoCalibrationKey, AutoCalibrationEntry>> result = new HashMap<>(fieldCount);
        for (int i = 0; i < fieldCount; i++) {
            String fieldName = in.readString();
            int entryCount = in.readVInt();
            Map<AutoCalibrationKey, AutoCalibrationEntry> entries = new HashMap<>(entryCount);
            for (int j = 0; j < entryCount; j++) {
                boolean calibrated = in.readBoolean();
                byte bits = in.readByte();
                byte queryBits = in.readByte();
                float oversample = in.readFloat();
                boolean precondition = in.readBoolean();
                int segmentCount = in.readVInt();
                long documentCount = in.readVLong();
                long sizeInBytes = in.readVLong();
                AutoCalibrationKey key = new AutoCalibrationKey(calibrated, bits, queryBits, oversample, precondition);
                entries.put(key, new AutoCalibrationEntry(key, segmentCount, documentCount, sizeInBytes));
            }
            result.put(fieldName, entries);
        }
        return result;
    }

    private void writeOptionalAutoCalibrationStats(StreamOutput out) throws IOException {
        if (autoCalibrationStats == null) {
            out.writeBoolean(false);
            return;
        }
        out.writeBoolean(true);
        out.writeVInt(autoCalibrationStats.size());
        for (var fieldEntry : autoCalibrationStats.entrySet()) {
            out.writeString(fieldEntry.getKey());
            out.writeVInt(fieldEntry.getValue().size());
            for (var entry : fieldEntry.getValue().values()) {
                out.writeBoolean(entry.key().calibrated());
                out.writeByte(entry.key().bits());
                out.writeByte(entry.key().queryBits());
                out.writeFloat(entry.key().oversample());
                out.writeBoolean(entry.key().precondition());
                out.writeVInt(entry.segmentCount());
                out.writeVLong(entry.documentCount());
                out.writeVLong(entry.sizeInBytes());
            }
        }
    }

    public void add(DenseVectorStats other) {
        if (other == null) {
            return;
        }
        this.valueCount += other.valueCount;
        if (other.offHeapStats != null) {
            if (this.offHeapStats == null) {
                this.offHeapStats = other.offHeapStats;
            } else {
                this.offHeapStats = Stream.of(this.offHeapStats, other.offHeapStats)
                    .flatMap(map -> map.entrySet().stream())
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, KnnVectorsReader::mergeOffHeapByteSizeMaps));
            }
        }
        if (other.autoCalibrationStats != null) {
            if (this.autoCalibrationStats == null) {
                this.autoCalibrationStats = other.autoCalibrationStats;
            } else {
                Map<String, Map<AutoCalibrationKey, AutoCalibrationEntry>> merged = new HashMap<>(this.autoCalibrationStats);
                for (var fieldEntry : other.autoCalibrationStats.entrySet()) {
                    merged.merge(fieldEntry.getKey(), fieldEntry.getValue(), DenseVectorStats::mergeAutoCalibrationEntries);
                }
                this.autoCalibrationStats = merged;
            }
        }
    }

    private static Map<AutoCalibrationKey, AutoCalibrationEntry> mergeAutoCalibrationEntries(
        Map<AutoCalibrationKey, AutoCalibrationEntry> a,
        Map<AutoCalibrationKey, AutoCalibrationEntry> b
    ) {
        Map<AutoCalibrationKey, AutoCalibrationEntry> result = new HashMap<>(a);
        for (var entry : b.entrySet()) {
            result.merge(entry.getKey(), entry.getValue(), AutoCalibrationEntry::merge);
        }
        return result;
    }

    /** Returns the total number of dense vectors added in the index. */
    public long getValueCount() {
        return valueCount;
    }

    /** Returns a map of per-field off-heap stats. */
    public Map<String, Map<String, Long>> offHeapStats() {
        return offHeapStats;
    }

    /** Returns per-field auto-calibration configuration groups with aggregated stats, or null if not collected. */
    public Map<String, Map<AutoCalibrationKey, AutoCalibrationEntry>> autoCalibrationStats() {
        return autoCalibrationStats;
    }

    private Map<String, Long> getTotalsByCategory() {
        if (offHeapStats == null) {
            return Map.of("veb", 0L, "vec", 0L, "veq", 0L, "vex", 0L, "cenivf", 0L, "clivf", 0L);
        } else {
            return offHeapStats.entrySet()
                .stream()
                .flatMap(map -> map.getValue().entrySet().stream())
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, Long::sum));
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(Fields.NAME);
        builder.field(Fields.VALUE_COUNT, valueCount);
        if (params.paramAsBoolean(INCLUDE_OFF_HEAP, false)) {
            toXContentWithFields(builder, params);
        }
        if (params.paramAsBoolean(INCLUDE_PER_FIELD_STATS, false) && autoCalibrationStats != null && autoCalibrationStats.isEmpty() == false) {
            toXContentAutoCalibration(builder);
        }
        builder.endObject();
        return builder;
    }

    private void toXContentWithFields(XContentBuilder builder, Params params) throws IOException {
        var totals = getTotalsByCategory();
        builder.startObject("off_heap");
        builder.humanReadableField("total_size_bytes", "total_size", ofBytes(totals.values().stream().mapToLong(Long::longValue).sum()));
        builder.humanReadableField("total_veb_size_bytes", "total_veb_size", ofBytes(totals.getOrDefault("veb", 0L)));
        builder.humanReadableField("total_vec_size_bytes", "total_vec_size", ofBytes(totals.getOrDefault("vec", 0L)));
        builder.humanReadableField("total_veq_size_bytes", "total_veq_size", ofBytes(totals.getOrDefault("veq", 0L)));
        builder.humanReadableField("total_vex_size_bytes", "total_vex_size", ofBytes(totals.getOrDefault("vex", 0L)));
        builder.humanReadableField("total_cenivf_size_bytes", "total_cenivf_size", ofBytes(totals.getOrDefault("cenivf", 0L)));
        builder.humanReadableField("total_clivf_size_bytes", "total_clivf_size", ofBytes(totals.getOrDefault("clivf", 0L)));
        if (params.paramAsBoolean(INCLUDE_PER_FIELD_STATS, false) && offHeapStats != null && offHeapStats.size() > 0) {
            toXContentWithPerFieldStats(builder);
        }
        builder.endObject();
    }

    private void toXContentAutoCalibration(XContentBuilder builder) throws IOException {
        builder.startObject(Fields.AUTO_CALIBRATION);
        for (var fieldName : autoCalibrationStats.keySet().stream().sorted().toList()) {
            builder.startArray(fieldName);
            List<AutoCalibrationEntry> entries = new ArrayList<>(autoCalibrationStats.get(fieldName).values());
            entries.sort(
                Comparator.comparing((AutoCalibrationEntry e) -> e.key().calibrated())
                    .thenComparing(e -> e.key().bits())
                    .thenComparing(e -> e.key().queryBits())
                    .thenComparing(e -> e.key().oversample())
                    .thenComparing(e -> e.key().precondition())
            );
            for (var entry : entries) {
                builder.startObject();
                builder.field(Fields.CALIBRATED, entry.key().calibrated());
                if (entry.key().calibrated()) {
                    builder.startObject(Fields.ENCODING);
                    builder.field(Fields.BITS, entry.key().bits());
                    builder.field(Fields.QUERY_BITS, entry.key().queryBits());
                    builder.endObject();
                    builder.field(Fields.OVERSAMPLE, entry.key().oversample());
                    builder.field(Fields.PRECONDITION, entry.key().precondition());
                }
                builder.field(Fields.SEGMENT_COUNT, entry.segmentCount());
                builder.field(Fields.DOCUMENT_COUNT, entry.documentCount());
                builder.field(Fields.SIZE_IN_BYTES, entry.sizeInBytes());
                builder.endObject();
            }
            builder.endArray();
        }
        builder.endObject();
    }

    private void toXContentWithPerFieldStats(XContentBuilder builder) throws IOException {
        builder.startObject(Fields.FIELDS);
        for (var key : offHeapStats.keySet().stream().sorted().toList()) {
            Map<String, Long> entry = offHeapStats.get(key);
            if (entry.isEmpty() == false) {
                builder.startObject(key);
                for (var eKey : entry.keySet().stream().sorted().toList()) {
                    long value = entry.get(eKey);
                    assert value >= 0L;
                    builder.humanReadableField(eKey + "_size_bytes", eKey + "_size", ofBytes(value));
                }
                builder.endObject();
            }
        }
        builder.endObject();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DenseVectorStats that = (DenseVectorStats) o;
        return valueCount == that.valueCount
            && Objects.equals(offHeapStats, that.offHeapStats)
            && Objects.equals(autoCalibrationStats, that.autoCalibrationStats);
    }

    @Override
    public int hashCode() {
        return Objects.hash(valueCount, offHeapStats, autoCalibrationStats);
    }

    public static final String INCLUDE_OFF_HEAP = "include_off_heap";
    public static final String INCLUDE_PER_FIELD_STATS = "include_per_field_stats";

    static final class Fields {
        static final String NAME = "dense_vector";
        static final String VALUE_COUNT = "value_count";
        static final String FIELDS = "fielddata";
        static final String AUTO_CALIBRATION = "auto_calibration";
        static final String CALIBRATED = "calibrated";
        static final String ENCODING = "encoding";
        static final String BITS = "bits";
        static final String QUERY_BITS = "query_bits";
        static final String OVERSAMPLE = "oversample";
        static final String PRECONDITION = "precondition";
        static final String SEGMENT_COUNT = "segment_count";
        static final String DOCUMENT_COUNT = "document_count";
        static final String SIZE_IN_BYTES = "size_in_bytes";
    }
}

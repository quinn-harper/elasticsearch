/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.mapper;

import org.elasticsearch.action.bulk.BulkRequestBuilder;
import org.elasticsearch.action.support.WriteRequest;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.test.ESSingleNodeTestCase;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertAcked;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertNoFailures;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertNoFailuresAndResponse;

/**
 * Indexes a flattened field with a mix of single-key, multi-key and very wide documents into a logsdb index across several segments,
 * then checks synthetic source before and after a force merge. Targets {@code NegativeArraySizeException} raised while reading keyed
 * flattened doc values, which happens when a keyed value has no separator byte.
 */
public class FlattenedFieldLogsdbSyntheticSourceMergeIT extends ESSingleNodeTestCase {

    private static final String INDEX = "logs-flattened-merge";
    private static final String ROOT = "cloudflare_logpush";
    private static final String MIDDLE = "device_posture";
    private static final String FLATTENED = "details";
    private static final int WIDE_DOCUMENT_KEY_COUNT = 3000;
    private static final int WIDE_DOCUMENT_VALUE_LENGTH = 200;

    private final Map<String, Map<String, TreeSet<String>>> expectedByDocumentId = new HashMap<>();

    public void testSyntheticSourceBeforeAndAfterMerge() throws IOException {
        createIndex();

        int numberOfRounds = between(5, 10);
        int nextId = 0;
        for (int round = 0; round < numberOfRounds; round++) {
            int documentsInRound = between(5, 40);
            BulkRequestBuilder bulk = client().prepareBulk().setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            for (int i = 0; i < documentsInRound; i++) {
                boolean overwriteExisting = expectedByDocumentId.isEmpty() == false && rarely();
                String id = overwriteExisting
                    ? randomFrom(new ArrayList<>(expectedByDocumentId.keySet()))
                    : Integer.toString(nextId++);
                bulk.add(prepareIndex(INDEX).setId(id).setSource(randomDocument(id)));
            }
            assertNoFailures(bulk.get());
            if (rarely() && expectedByDocumentId.isEmpty() == false) {
                String idToDelete = randomFrom(new ArrayList<>(expectedByDocumentId.keySet()));
                client().prepareDelete(INDEX, idToDelete).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE).get();
                expectedByDocumentId.remove(idToDelete);
            }
        }

        assertAllDocumentsReadable();

        assertNoFailures(indicesAdmin().prepareForceMerge(INDEX).setMaxNumSegments(1).get());
        indicesAdmin().prepareRefresh(INDEX).get();

        assertAllDocumentsReadable();
    }

    private void createIndex() throws IOException {
        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject("@timestamp")
            .field("type", "date")
            .endObject()
            .startObject("host")
            .startObject("properties")
            .startObject("name")
            .field("type", "keyword")
            .endObject()
            .endObject()
            .endObject()
            .startObject(ROOT)
            .startObject("properties")
            .startObject(MIDDLE)
            .startObject("properties")
            .startObject(FLATTENED)
            .field("type", "flattened");
        if (randomBoolean()) {
            mapping.field("ignore_above", between(5, 40));
        }
        mapping.endObject().endObject().endObject().endObject().endObject().endObject().endObject();

        Settings.Builder settings = Settings.builder().put("index.mode", "logsdb").put("index.number_of_shards", 1);
        if (randomBoolean()) {
            settings.put("index.mapping.synthetic_source_keep", "arrays");
        }
        assertAcked(indicesAdmin().prepareCreate(INDEX).setSettings(settings).setMapping(mapping));
    }

    private XContentBuilder randomDocument(String id) throws IOException {
        Map<String, TreeSet<String>> expected = new HashMap<>();
        int shape = between(0, 9);
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field("@timestamp", 1_700_000_000_000L + randomLongBetween(0, 1_000_000L))
            .startObject("host")
            .field("name", "host-" + between(0, 3))
            .endObject()
            .startObject(ROOT)
            .startObject(MIDDLE)
            .startObject(FLATTENED);
        if (shape < 4) {
            addKey(builder, expected, "k" + between(0, 5), randomAlphaOfLengthBetween(0, 12));
        } else if (shape < 8) {
            int keyCount = between(2, 12);
            for (int i = 0; i < keyCount; i++) {
                String key = "k" + i;
                if (randomBoolean()) {
                    addKey(builder, expected, key, randomUnicodeOfLengthBetween(0, 30));
                } else {
                    int valueCount = between(2, 5);
                    List<String> values = new ArrayList<>();
                    for (int v = 0; v < valueCount; v++) {
                        values.add(randomBoolean() ? "dup" : randomAlphaOfLengthBetween(1, 20));
                    }
                    builder.startArray(key);
                    for (String value : values) {
                        builder.value(value);
                    }
                    builder.endArray();
                    expected.computeIfAbsent(key, k -> new TreeSet<>()).addAll(values);
                }
            }
        } else {
            for (int i = 0; i < WIDE_DOCUMENT_KEY_COUNT; i++) {
                addKey(builder, expected, "wide" + i, randomAlphaOfLength(WIDE_DOCUMENT_VALUE_LENGTH));
            }
        }
        builder.endObject().endObject().endObject().endObject();
        expectedByDocumentId.put(id, expected);
        return builder;
    }

    private static void addKey(XContentBuilder builder, Map<String, TreeSet<String>> expected, String key, String value)
        throws IOException {
        builder.field(key, value);
        expected.computeIfAbsent(key, k -> new TreeSet<>()).add(value);
    }

    @SuppressWarnings("unchecked")
    private void assertAllDocumentsReadable() {
        assertNoFailuresAndResponse(
            client().prepareSearch(INDEX).setQuery(QueryBuilders.matchAllQuery()).setSize(expectedByDocumentId.size() + 10),
            response -> {
                assertEquals(expectedByDocumentId.size(), response.getHits().getHits().length);
                for (SearchHit hit : response.getHits().getHits()) {
                    Map<String, Object> source = hit.getSourceAsMap();
                    Map<String, Object> root = (Map<String, Object>) source.get(ROOT);
                    Map<String, Object> middle = (Map<String, Object>) root.get(MIDDLE);
                    Map<String, Object> details = (Map<String, Object>) middle.get(FLATTENED);
                    Map<String, TreeSet<String>> expected = expectedByDocumentId.get(hit.getId());
                    assertEquals("keys of document [" + hit.getId() + "]", expected.keySet(), details.keySet());
                    for (Map.Entry<String, TreeSet<String>> entry : expected.entrySet()) {
                        Object actual = details.get(entry.getKey());
                        TreeSet<String> actualValues = new TreeSet<>();
                        if (actual instanceof List<?> list) {
                            for (Object element : list) {
                                actualValues.add(element == null ? null : element.toString());
                            }
                        } else {
                            actualValues.add(actual.toString());
                        }
                        assertEquals(
                            "values of key [" + entry.getKey() + "] of document [" + hit.getId() + "]",
                            entry.getValue(),
                            actualValues
                        );
                    }
                }
            }
        );
    }
}

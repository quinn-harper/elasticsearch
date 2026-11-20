/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.query;

import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.index.query.SpanNearQueryBuilder;
import org.elasticsearch.test.ESSingleNodeTestCase;

import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertHitCount;

/**
 * Reproduces a {@code NullPointerException: Cannot invoke "CollectionStatistics.field()" because "collectionStats" is null}
 * seen on serverless search nodes.
 *
 * <p>{@code SpanWeight} takes term statistics from the terms inside the span and collection statistics from the field the
 * span reports. A {@code field_masking_span} makes those two fields differ. When the reported field has no indexed terms,
 * {@code IndexSearcher#collectionStatistics} returns null while the masked terms still have a document frequency, and the
 * per field similarity dereferences the null statistics.
 */
public class SpanFieldMaskingSearchTests extends ESSingleNodeTestCase {

    private static final String INDEX = "span-field-masking";

    public void testReportedFieldMappedButWithoutTerms() {
        createIndex(INDEX, Settings.EMPTY, "a", "type=text", "b", "type=text");
        prepareIndex(INDEX).setSource("b", "foo bar").get();
        client().admin().indices().prepareRefresh(INDEX).get();

        assertHitCount(client().prepareSearch(INDEX).setQuery(maskedSpanNear()), 1);
    }

    public void testReportedFieldUnmapped() {
        createIndex(INDEX, Settings.EMPTY, "b", "type=text");
        prepareIndex(INDEX).setSource("b", "foo bar").get();
        client().admin().indices().prepareRefresh(INDEX).get();

        assertHitCount(client().prepareSearch(INDEX).setQuery(maskedSpanNear()), 1);
    }

    public void testReportedFieldWithTermsControl() {
        createIndex(INDEX, Settings.EMPTY, "a", "type=text", "b", "type=text");
        prepareIndex(INDEX).setSource("a", "something", "b", "foo bar").get();
        client().admin().indices().prepareRefresh(INDEX).get();

        assertHitCount(client().prepareSearch(INDEX).setQuery(maskedSpanNear()), 1);
    }

    private static QueryBuilder maskedSpanNear() {
        SpanNearQueryBuilder spanNear = QueryBuilders.spanNearQuery(
            QueryBuilders.fieldMaskingSpanQuery(QueryBuilders.spanTermQuery("b", "foo"), "a"),
            0
        ).addClause(QueryBuilders.fieldMaskingSpanQuery(QueryBuilders.spanTermQuery("b", "bar"), "a")).inOrder(true);
        return QueryBuilders.boolQuery().must(spanNear);
    }
}

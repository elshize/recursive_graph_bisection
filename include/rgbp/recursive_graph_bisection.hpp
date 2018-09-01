#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <gsl/span>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <rgbp/util.hpp>

namespace constants {
const int MAX_ITER = 20;
const uint64_t PARALLEL_SWITCH_DEPTH = 6;
}

struct docid_node {
    uint64_t initial_id;
    gsl::span<uint32_t> terms_;
    gsl::span<uint32_t> freqs;
    size_t num_terms;
    size_t num_terms_not_pruned;
    std::vector<uint32_t> terms() const
    {
        return std::vector<uint32_t>(terms_.begin(), terms_.end());
    }
};

std::vector<float> log2_precomp;

float log2_cmp(uint32_t idx)
{
    if (idx < 256)
        return log2_precomp[idx];
    return log2f(idx);
}

void swap_nodes(docid_node* a, docid_node* b)
{
    std::swap(a->initial_id, b->initial_id);
    std::swap(a->terms_, b->terms_);
    std::swap(a->freqs, b->freqs);
    std::swap(a->num_terms, b->num_terms);
    std::swap(a->num_terms_not_pruned, b->num_terms_not_pruned);
}

void swap_nodes(docid_node* a, docid_node* b, std::vector<uint32_t>& deg1,
    std::vector<uint32_t>& deg2, std::vector<uint8_t>& queries_changed)
{
    auto terms = a->terms();
    for (size_t i = 0; i < a->num_terms; i++) {
        auto qry = terms[i];
        deg1[qry]--;
        deg2[qry]++;
        queries_changed[qry] = 1;
    }

    terms = b->terms();
    for (size_t i = 0; i < b->num_terms; i++) {
        auto qry = terms[i];
        deg1[qry]++;
        deg2[qry]--;
        queries_changed[qry] = 1;
    }
    swap_nodes(a, b);
}

struct bipartite_graph {
    size_t num_queries;
    size_t num_docs;
    size_t num_docs_inc_empty;
    std::vector<docid_node> graph;
    std::vector<uint32_t> doc_contents;
    std::vector<uint32_t> doc_freqs;
};

struct partition_t {
    docid_node* V1;
    docid_node* V2;
    size_t n1;
    size_t n2;
};

void compute_doc_sizes(const inverted_index& idx, std::vector<uint32_t>& doc_sizes,
    std::vector<uint32_t>& doc_sizes_non_pruned, uint32_t min_doc_id,
    uint32_t max_doc_id, size_t min_list_len)
{
    for (size_t termid = 0; termid < idx.docids.size(); termid++) {
        const auto& plist = idx.docids[termid];
        for (const auto& doc_id : plist) {
            if (min_doc_id <= doc_id && doc_id < max_doc_id) {
                if (plist.size() >= min_list_len) {
                    doc_sizes[doc_id] += 1;
                }
                doc_sizes_non_pruned[doc_id] += 1;
            }
        }
    }
}

void create_graph(bipartite_graph& bg, const inverted_index& idx, uint32_t min_doc_id,
    uint32_t max_doc_id, size_t min_list_len)
{
    std::vector<uint32_t> doc_offset(idx.max_doc_id + 1, 0);
    for (size_t termid = 0; termid < idx.docids.size(); termid++) {
        const auto& dlist = idx.docids[termid];
        const auto& flist = idx.freqs[termid];
        if (dlist.size() >= min_list_len) {
            for (size_t pos = 0; pos < dlist.size(); pos++) {
                const auto& doc_id = dlist[pos];
                if (min_doc_id <= doc_id && doc_id < max_doc_id) {
                    bg.graph[doc_id].initial_id = doc_id;
                    bg.graph[doc_id].freqs[doc_offset[doc_id]] = flist[pos];
                    bg.graph[doc_id].terms_[doc_offset[doc_id]++] = termid;
                }
            }
        }
    }
    for (size_t termid = 0; termid < idx.docids.size(); termid++) {
        const auto& dlist = idx.docids[termid];
        const auto& flist = idx.freqs[termid];
        if (dlist.size() < min_list_len) {
            for (size_t pos = 0; pos < dlist.size(); pos++) {
                const auto& doc_id = dlist[pos];
                if (min_doc_id <= doc_id && doc_id < max_doc_id) {
                    bg.graph[doc_id].initial_id = doc_id;
                    bg.graph[doc_id].freqs[doc_offset[doc_id]] = flist[pos];
                    bg.graph[doc_id].terms_[doc_offset[doc_id]++] = termid;
                }
            }
        }
    }
}

struct compute_doc_sizes_par {
    const inverted_index& idx;
    size_t min_list_len;
    std::vector<uint32_t>& doc_sizes;
    std::vector<uint32_t>& doc_sizes_non_pruned;
    compute_doc_sizes_par(const inverted_index& idx, size_t min_list_len,
        std::vector<uint32_t>& doc_sizes,
        std::vector<uint32_t>& doc_sizes_non_pruned)
        : idx(idx)
        , min_list_len(min_list_len)
        , doc_sizes(doc_sizes)
        , doc_sizes_non_pruned(doc_sizes_non_pruned)
    {}
    void operator()(const tbb::blocked_range<uint32_t>& r) const
    {
        for (size_t termid = 0; termid < idx.docids.size(); termid++) {
            const auto& plist = idx.docids[termid];
            for (const auto& doc_id : plist) {
                if (r.begin() <= doc_id && doc_id < r.end()) {
                    if (plist.size() >= min_list_len) {
                        doc_sizes[doc_id] += 1;
                    }
                    doc_sizes_non_pruned[doc_id] += 1;
                }
            }
        }
    }
};

void init_bg(
    const inverted_index& idx, size_t min_list_len, bipartite_graph& bg)
{
    timer t("init sizes");
    std::vector<uint32_t> doc_sizes(idx.max_doc_id + 1, 0u);
    std::vector<uint32_t> doc_sizes_non_pruned(idx.max_doc_id + 1, 0u);
    compute_doc_sizes_par sizes(
        idx, min_list_len, doc_sizes, doc_sizes_non_pruned);
    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0, idx.max_doc_id + 1), sizes);
    bg.doc_contents.resize(idx.num_postings);
    bg.doc_freqs.resize(idx.num_postings);
    bg.graph.resize(idx.max_doc_id + 1);
    bg.num_docs_inc_empty = idx.max_doc_id + 1;

    size_t offset = 0;
    for (size_t i = 0; i < doc_sizes.size(); i++) {
        bg.graph[i].terms_ = gsl::make_span(
            bg.doc_contents.data() + offset, doc_sizes_non_pruned[i]);
        bg.graph[i].freqs = gsl::make_span(
            bg.doc_freqs.data() + offset, doc_sizes_non_pruned[i]);
        bg.graph[i].num_terms = doc_sizes[i];
        bg.graph[i].num_terms_not_pruned = doc_sizes_non_pruned[i];
        offset += doc_sizes_non_pruned[i];
    }
    // Set ID for empty documents.
    //for (uint32_t doc_id = 0; doc_id < idx.num_docs; ++doc_id) {
    //    if (bg.graph[doc_id].initial_id != doc_id) {
    //        bg.graph[doc_id].initial_id = doc_id;
    //    }
    //}
}

void create_forward_index(
    const inverted_index& idx, size_t min_list_len, bipartite_graph& bg)
{
    timer t("create forward index");
    tbb::parallel_for(tbb::blocked_range<size_t>(0, idx.num_docs),
        [&](const auto& range) {
            create_graph(bg, idx, range.begin(), range.end(), min_list_len);
        });
}

size_t remove_empty(bipartite_graph& bg)
{
    size_t num_empty = 0;
    {
        // all docs with 0 size go to the back!
        auto empty_cmp = [](const auto& a, const auto& b) {
            return a.num_terms > b.num_terms;
        };
        std::sort(bg.graph.begin(), bg.graph.end(), empty_cmp);
        auto last_non_empty = std::find_if(bg.graph.rbegin(), bg.graph.rend(),
            [](const auto& node) { return node.num_terms > 0; });
        num_empty = std::distance(bg.graph.rbegin(), last_non_empty);
        bg.num_docs = bg.num_docs_inc_empty - num_empty;
    }
    return num_empty;
}

std::pair<size_t, size_t> calc_skipped_lists(
    const inverted_index& idx, size_t min_list_len)
{
    size_t num_skipped_lists = 0;
    size_t num_lists = 0;
    for (size_t termid = 0; termid < idx.docids.size(); termid++) {
        const auto& dlist = idx.docids[termid];
        if (dlist.size() < min_list_len) {
            num_skipped_lists++;
        } else {
            num_lists++;
        }
    }
    return {num_skipped_lists, num_lists};
}

bipartite_graph construct_bipartite_graph(
    inverted_index& idx, size_t min_list_len)
{
    timer t("construct_bipartite_graph");
    bipartite_graph bg;
    bg.num_queries = idx.size();
    init_bg(idx, min_list_len, bg);
    create_forward_index(idx, min_list_len, bg);
    size_t num_empty_docs = remove_empty(bg);
    auto [num_skipped_lists, num_lists] = calc_skipped_lists(idx, min_list_len);
    std::cout << "\tnum_empty docs = " << num_empty_docs << std::endl;
    std::cout << "\tnum_skipped lists = " << num_skipped_lists << std::endl;
    std::cout << "\tnum_lists = " << num_lists << std::endl;
    std::cout << "\tnum_docs = " << bg.num_docs << std::endl;
    return bg;
}

void recreate_lists(const bipartite_graph& bg, inverted_index& idx,
    uint32_t min_q_id, uint32_t max_q_id, std::vector<uint32_t>& qmap,
    std::vector<uint32_t>& dsizes)
{
    for (size_t docid = 0; docid < bg.num_docs_inc_empty; docid++) {
        const auto& doc = bg.graph[docid];
        auto terms = doc.terms();
        for (size_t i = 0; i < doc.num_terms_not_pruned; i++) {
            auto qid = terms[i];
            if (min_q_id <= qmap[qid] && qmap[qid] < max_q_id) {
                auto freq = doc.freqs[i];
                idx.docids[qid].push_back(docid);
                idx.freqs[qid].push_back(freq);
                dsizes[docid] += freq;
            }
        }
    }
}

inverted_index recreate_invidx(const bipartite_graph& bg, size_t num_lists)
{
    timer t("recreate_invidx");
    inverted_index idx;
    size_t num_postings = 0;
    idx.resize(num_lists);
    {
        size_t workers = 1;
        size_t qids_in_slice = num_lists / workers;
        std::vector<uint32_t> qids_map(num_lists);
        for (size_t i = 0; i < qids_map.size(); i++)
            qids_map[i] = i;
        std::mt19937 rnd(1);
        std::shuffle(qids_map.begin(), qids_map.end(), rnd);
        std::vector<std::vector<uint32_t>> doc_sizes(workers);
        for (size_t id = 0; id < workers; id++) {
            doc_sizes[id].resize(bg.num_docs_inc_empty);
            size_t min_q_id = id * qids_in_slice;
            size_t max_q_id = min_q_id + qids_in_slice;
            max_q_id = num_lists;
            recreate_lists(
                bg, idx, min_q_id, max_q_id, qids_map, doc_sizes[id]);
        }
        idx.doc_lengths.resize(bg.num_docs_inc_empty);
        for (size_t id = 0; id < workers; id++) {
            for (size_t docid = 0; docid < bg.num_docs_inc_empty; docid++) {
                idx.doc_lengths[docid] += doc_sizes[id][docid];
            }
        }
    }
    {

        for (size_t docid = 0; docid < bg.num_docs_inc_empty; docid++) {
            const auto& doc = bg.graph[docid];
            idx.doc_id_mapping.push_back(doc.initial_id);
            num_postings += doc.num_terms_not_pruned;
        }
    }
    idx.num_docs = bg.num_docs_inc_empty;
    idx.max_doc_id = idx.num_docs - 1;
    idx.num_postings = num_postings;
    std::cout << "\tnum_docs = " << idx.num_docs << std::endl;
    std::cout << "\tmax_doc_id = " << idx.max_doc_id << std::endl;
    std::cout << "\tnum_lists = " << idx.docids.size() << std::endl;
    std::cout << "\tnum_postings = " << idx.num_postings << std::endl;
    return idx;
}

/* random shuffle seems to do ok */
partition_t initial_partition(docid_node* G, size_t n)
{
    partition_t p;
    std::mt19937 rnd(n);
    std::shuffle(G, G + n, rnd);
    p.V1 = G;
    p.n1 = (n / 2);
    p.V2 = G + p.n1;
    p.n2 = n - p.n1;
    return p;
}

struct move_gain {
    double gain;
    docid_node* node;
    move_gain()
        : gain(0)
        , node(nullptr)
    {
    }
    move_gain(double g, docid_node* n)
        : gain(g)
        , node(n)
    {
    }
    bool operator<(const move_gain& other) { return gain > other.gain; }
};

struct move_gains_t {
    std::vector<move_gain> V1;
    std::vector<move_gain> V2;
};

move_gain compute_single_gain(
    docid_node* doc, std::vector<float>& before, std::vector<float>& after)
{
    float before_move = 0.0;
    float after_move = 0.0;
    auto terms = doc->terms();
    for (size_t j = 0; j < doc->num_terms; j++) {
        auto q = terms[j];
        before_move += before[q];
        after_move += after[q];
    }
    float gain = before_move - after_move;
    return move_gain(gain, doc);
}

void compute_deg(docid_node* docs, size_t n, std::vector<uint32_t>& deg)
{
    for (size_t i = 0; i < n; i++) {
        auto doc = docs + i;
        auto terms = doc->terms();
        for (size_t j = 0; j < doc->num_terms; j++) {
            deg[terms[j]]++;
        }
    }
}

void compute_gains(docid_node* docs, size_t n, std::vector<float>& before,
    std::vector<float>& after, std::vector<move_gain>& res)
{
    res.resize(n);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const auto& block) {
        for (size_t idx = block.begin(); idx != block.end(); ++idx) {
            auto doc = docs + idx;
            res[idx] = compute_single_gain(doc, before, after);
        }
    });
}

void compute_gains_np(docid_node* docs, size_t n, std::vector<float>& before,
    std::vector<float>& after, std::vector<move_gain>& res)
{
    res.resize(n);
    for (size_t i = 0; i < n; i++) {
        auto doc = docs + i;
        res[i] = compute_single_gain(doc, before, after);
    }
}

struct compute_move_gains_range {
    std::vector<uint32_t>& deg1;
    std::vector<uint32_t>& deg2;
    std::vector<float>& before;
    std::vector<float>& left2right;
    std::vector<float>& right2left;
    std::vector<uint8_t>& qry_changed;
    float logn1;
    float logn2;
    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t q = range.begin(); q != range.end(); ++q) {
            if (qry_changed[q] == 1) {
                qry_changed[q] = 0;
                before[q] = 0;
                left2right[q] = 0;
                right2left[q] = 0;
                if (deg1[q] or deg2[q]) {
                    before[q] = deg1[q] * logn1
                        - deg1[q] * log2_cmp(deg1[q] + 1) + deg2[q] * logn2
                        - deg2[q] * log2_cmp(deg2[q] + 1);
                }
                if (deg1[q]) {
                    left2right[q] = (deg1[q] - 1) * logn1
                        - (deg1[q] - 1) * log2_cmp(deg1[q])
                        + (deg2[q] + 1) * logn2
                        - (deg2[q] + 1) * log2_cmp(deg2[q] + 2);
                }
                if (deg2[q])
                    right2left[q] = (deg1[q] + 1) * logn1
                        - (deg1[q] + 1) * log2_cmp(deg1[q] + 2)
                        + (deg2[q] - 1) * logn2
                        - (deg2[q] - 1) * log2_cmp(deg2[q]);
            }
        }
    }
};

move_gains_t compute_move_gains(partition_t& P, size_t num_queries,
    std::vector<uint32_t>& deg1, std::vector<uint32_t>& deg2,
    std::vector<float>& before, std::vector<float>& left2right,
    std::vector<float>& right2left, std::vector<uint8_t>& qry_changed)
{
    move_gains_t gains;

    float logn1 = log2f(P.n1);
    float logn2 = log2f(P.n2);
    compute_move_gains_range f{ deg1, deg2, before, left2right, right2left,
        qry_changed, logn1, logn2 };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_queries), f);

    // (2) compute gains from moving docs
    // TODO: parallel
    compute_gains(P.V1, P.n1, before, left2right, gains.V1);
    compute_gains(P.V2, P.n2, before, right2left, gains.V2);

    return gains;
}

move_gains_t compute_move_gains_np(partition_t& P, size_t num_queries,
    std::vector<uint32_t>& deg1, std::vector<uint32_t>& deg2,
    std::vector<float>& before, std::vector<float>& left2right,
    std::vector<float>& right2left, std::vector<uint8_t>& qry_changed)
{
    move_gains_t gains;

    float logn1 = log2f(P.n1);
    float logn2 = log2f(P.n2);
    for (size_t q = 0; q < num_queries; q++) {
        if (qry_changed[q] == 1) {
            qry_changed[q] = 0;
            before[q] = 0;
            left2right[q] = 0;
            right2left[q] = 0;
            if (deg1[q] or deg2[q]) {
                before[q] = deg1[q] * logn1 - deg1[q] * log2_cmp(deg1[q] + 1)
                    + deg2[q] * logn2 - deg2[q] * log2_cmp(deg2[q] + 1);
            }
            if (deg1[q]) {
                left2right[q] = (deg1[q] - 1) * logn1
                    - (deg1[q] - 1) * log2_cmp(deg1[q]) + (deg2[q] + 1) * logn2
                    - (deg2[q] + 1) * log2_cmp(deg2[q] + 2);
            }
            if (deg2[q])
                right2left[q] = (deg1[q] + 1) * logn1
                    - (deg1[q] + 1) * log2_cmp(deg1[q] + 2)
                    + (deg2[q] - 1) * logn2 - (deg2[q] - 1) * log2_cmp(deg2[q]);
        }
    }

    // (2) compute gains from moving docs
    compute_gains(P.V1, P.n1, before, left2right, gains.V1);
    compute_gains(P.V2, P.n2, before, right2left, gains.V2);

    return gains;
}

void recursive_bisection_np(progress_bar& progress, docid_node* G,
    size_t num_queries, size_t n,uint64_t depth,uint64_t max_depth)
{
    // (1) create the initial partition. O(n)
    auto partition = initial_partition(G, n);

    {
        // (2) we compute deg1 and deg2 only once
        std::vector<uint32_t> deg1(num_queries, 0);
        std::vector<uint32_t> deg2(num_queries, 0);
        std::vector<float> before(num_queries);
        std::vector<float> left2right(num_queries);
        std::vector<float> right2left(num_queries);

        std::vector<uint8_t> query_changed(num_queries, 1);
        {
            compute_deg(partition.V1, partition.n1, deg1);
            compute_deg(partition.V2, partition.n2, deg2);
        }

        // (3) perform bisection. constant number of iterations
        for (int cur_iter = 1; cur_iter <= constants::MAX_ITER; cur_iter++) {
            // (3a) compute move gains
            auto gains = compute_move_gains_np(partition, num_queries, deg1,
                deg2, before, left2right, right2left, query_changed);
            memset(query_changed.data(), 0, num_queries);

            // (3b) sort by decreasing gain. O(n log n)
            {
                std::sort(gains.V1.begin(), gains.V1.end());
                std::sort(gains.V2.begin(), gains.V2.end());
            }

            // (3c) swap. O(n)
            size_t num_swaps = 0;
            {
                auto itr_v1 = gains.V1.begin();
                auto itr_v2 = gains.V2.begin();
                while (itr_v1 != gains.V1.end() && itr_v2 != gains.V2.end()) {
                    if (itr_v1->gain + itr_v2->gain > 0) {
                        // maybe we need to do something here to make
                        // compute_move_gains() efficient?
                        swap_nodes(itr_v1->node, itr_v2->node, deg1, deg2,
                            query_changed);
                        num_swaps++;
                    } else {
                        break;
                    }
                    ++itr_v1;
                    ++itr_v2;
                }
            }

            // (3d) converged?
            if (num_swaps == 0) {
                break;
            }
        }
    }

    // (4) recurse. at most O(log n) recursion steps
    if (depth + 1 <= max_depth) {
        if (partition.n1 > 1)
            recursive_bisection_np(
                progress, partition.V1, num_queries, partition.n1, depth + 1,max_depth);
        if (partition.n2 > 1)
            recursive_bisection_np(
                progress, partition.V2, num_queries, partition.n2, depth + 1,max_depth);

        if (partition.n1 == 1)
            progress.done(1);
        if (partition.n2 == 1)
            progress.done(1);
    } else {
        progress.done(n);
    }
}

void recursive_bisection(progress_bar& progress, docid_node* G,
    size_t num_queries, size_t n, uint64_t depth,uint64_t max_depth)
{
    // (1) create the initial partition. O(n)
    auto partition = initial_partition(G, n);

    {
        // (2) we compute deg1 and deg2 only once
        std::vector<uint32_t> deg1(num_queries, 0);
        std::vector<uint32_t> deg2(num_queries, 0);
        std::vector<float> before(num_queries);
        std::vector<float> left2right(num_queries);
        std::vector<float> right2left(num_queries);

        std::vector<uint8_t> query_changed(num_queries, 1);
        {
            // TODO: parallel
            compute_deg(partition.V1, partition.n1, deg1);
            compute_deg(partition.V2, partition.n2, deg2);
        }

        // (3) perform bisection. constant number of iterations
        for (int cur_iter = 1; cur_iter <= constants::MAX_ITER; cur_iter++) {
            // (3a) compute move gains
            auto gains = compute_move_gains(partition, num_queries, deg1, deg2,
                before, left2right, right2left, query_changed);
            memset(query_changed.data(), 0, num_queries);

            // (3b) sort by decreasing gain. O(n log n)
            {
                // TODO: parallel
                std::sort(gains.V1.begin(), gains.V1.end());
                std::sort(gains.V2.begin(), gains.V2.end());
            }

            // (3c) swap. O(n)
            size_t num_swaps = 0;
            {
                auto itr_v1 = gains.V1.begin();
                auto itr_v2 = gains.V2.begin();
                while (itr_v1 != gains.V1.end() && itr_v2 != gains.V2.end()) {
                    if (itr_v1->gain + itr_v2->gain > 0) {
                        // maybe we need to do something here to make
                        // compute_move_gains() efficient?
                        swap_nodes(itr_v1->node, itr_v2->node, deg1, deg2,
                            query_changed);
                        num_swaps++;
                    } else {
                        break;
                    }
                    ++itr_v1;
                    ++itr_v2;
                }
            }

            // (3d) converged?
            if (num_swaps == 0) {
                break;
            }
        }
    }

    // (4) recurse. at most O(log n) recursion steps
    if (depth + 1 <= max_depth) {
        if (depth < constants::PARALLEL_SWITCH_DEPTH) {
            if (partition.n1 > 1) {
                recursive_bisection(progress, partition.V1,
                    num_queries, partition.n1, depth + 1,max_depth);
            }
            if (partition.n2 > 1) {
                recursive_bisection(progress, partition.V2, num_queries,
                    partition.n2, depth + 1,max_depth);
            }
        } else {
            if (partition.n1 > 1) {
                recursive_bisection_np(progress, partition.V1, num_queries,
                    partition.n1, depth + 1,max_depth);
            }
            if (partition.n2 > 1) {
                recursive_bisection_np(progress, partition.V2, num_queries,
                    partition.n2, depth + 1,max_depth);
            }
        }
        if (partition.n1 == 1)
            progress.done(1);
        if (partition.n2 == 1)
            progress.done(1);
    } else {
        progress.done(n);
    }
}

inverted_index reorder_docids_graph_bisection(
    inverted_index& invidx, size_t min_list_len)
{
    auto num_lists = invidx.docids.size();
    auto bg = construct_bipartite_graph(invidx, min_list_len);

    // free up some space
    invidx.clear();

    // make things faster by precomputing some logs
    log2_precomp.resize(256);
    for(size_t i = 0; i < 256; i++) { log2_precomp[i] = log2f(i); }

    {
        auto max_depth = std::max(1.0,ceil(log2(bg.num_docs)-5));
        std::cout << "recursion depth = " << max_depth << std::endl;
        timer t("recursive_bisection");
        progress_bar bp("recursive_bisection", bg.num_docs);
        recursive_bisection(bp, bg.graph.data(), bg.num_queries, bg.num_docs, 0, max_depth);
    }
    return recreate_invidx(bg, num_lists);
}

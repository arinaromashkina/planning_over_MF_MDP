# Datasets

| Dataset | Domain | Positive feedback | Catalog | Access |
|---|---|---|---|---|
| MovieLens-1M | movies | rating >= 4 | native (~3706 items) | public, files.grouplens.org |
| KuaiRec | short video | high watch ratio | native | public |
| KuaiRand | short video (randomized exposure) | — | — | public; inspect before use |
| VK-LSVD (VK_IP / VK_UP) | short video (industrial) | — | top-5000 pool | restricted; check access |
| YAMbDA | music | played ratio >= 80% | top-5000 pool | restricted; check access |

Order: MovieLens-1M first (reproduction + debugging), then KuaiRec, then KuaiRand inspection.
Do not commit raw data. Record source URL, version/date, and checksum on download.

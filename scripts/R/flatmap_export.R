# Export the atlas segment mask (Huang et al. 2026, CNGB STDS0000395: bin50_segment_mask_combined_preprocess.rds = their
# data/bin50_segment.rds) for the sections of our series that the atlas covers (T408-T428): one row per bin50 pixel.
# -> results/flatmap/mask_T408_T428.csv.gz (slice, x, y, id)
x <- readRDS('/workspace/external_repos/mccsta_data/bin50_segment_mask_combined_preprocess.rds')
s <- as.integer(sub('T', '', x$slice)); x <- x[s >= 408 & s <= 428, c('slice', 'x', 'y', 'id')]
cat('rows', nrow(x), '| sections', length(unique(x$slice)), '| segments', length(unique(x$id)), '\n')
g <- gzfile('/workspace/Gene_modules/benchmark_alignment_2026_10/results/flatmap/mask_T408_T428.csv.gz', 'w'); write.csv(x, g, row.names = FALSE); close(g)

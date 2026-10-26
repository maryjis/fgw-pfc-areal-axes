# Export the atlas segment x gene matrix (Huang et al. 2026, Segment_genes_marmoset.rds, slot `data`) for the 5 000 variable
# features used in their PCA, all 17 525 segments: float32 row-major matrix segments x genes plus name lists.
# -> results/segments/atlas_data_hvg5000.f32, atlas_data_hvg5000_genes.txt, atlas_data_hvg5000_segments.txt
suppressMessages({library(SeuratObject); library(Matrix)})
O <- '/workspace/Gene_modules/benchmark_alignment_2026_10/results/segments'
x <- readRDS('/workspace/Segment_genes_marmoset.rds'); g <- VariableFeatures(x); d <- x@assays$RNA@data[g, ]
m <- t(as.matrix(d)); storage.mode(m) <- 'double'
cat('matrix', dim(m), '| range', range(m), '\n')
con <- file(file.path(O, 'atlas_data_hvg5000.f32'), 'wb'); writeBin(as.vector(t(m)), con, size = 4); close(con)   # row-major: segment by segment
writeLines(colnames(m), file.path(O, 'atlas_data_hvg5000_genes.txt')); writeLines(rownames(m), file.path(O, 'atlas_data_hvg5000_segments.txt'))

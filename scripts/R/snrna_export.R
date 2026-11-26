#!/usr/bin/env Rscript
# Export raw counts of two excitatory groups from the snRNA-seq Seurat object (preprocess_ctx) for the gene-set axis scores.
#   L2/3 IT group : L2/3 IT + ExN_dev_1 + ExN_dev_2        L4 IT group : L4 IT + L4 IT-V1
# The subclass comes from the final metadata (8f_metarefine_all_metadata.csv, which replaces meta.data per the README).
# Counts: the raw integer layers counts.1/2/3 joined over cells, ALL genes (the CP10k denominator is the nucleus total).
# usage: Rscript snrna_export.R <seurat ExN rds> <8f metadata csv> <out dir> [age]
#   with [age] (e.g. adult, on the All object): every nucleus of that age, all subclasses; group = the merged group for L2/3 IT and
#   L4 IT as above, otherwise the subclass itself (input of ottr.genescore.snrna_types, out dir work/snrna_types)
# -> <out>/counts_{i,p,x}.bin (the CSC slots of the genes x nuclei dgCMatrix: int32 row indices, int32 column pointers, int32
#    counts), shape.txt (genes nuclei nnz), genes.txt, cells.tsv (cell, samplename, age, donor, region, region_label, subclass, group, n_count)
suppressMessages({ library(SeuratObject); library(Matrix) })
a <- commandArgs(trailingOnly = TRUE); rds <- a[[1]]; meta_f <- a[[2]]; out <- a[[3]]; only_age <- if (length(a) > 3) a[[4]] else NULL
dir.create(out, showWarnings = FALSE, recursive = TRUE)
GROUPS <- list("L2/3 IT" = c("L2/3 IT", "ExN_dev_1", "ExN_dev_2"), "L4 IT" = c("L4 IT", "L4 IT-V1"))

o <- readRDS(rds); md <- o@meta.data
f <- read.csv(meta_f, stringsAsFactors = FALSE); f <- f[match(rownames(md), f$seurat_cell_id), ]
if (anyNA(f$seurat_cell_id)) stop("cells of the object missing in the final metadata: ", sum(is.na(f$seurat_cell_id)))
md$subclass <- f$subclass
grp <- rep(NA_character_, nrow(md)); for (g in names(GROUPS)) grp[md$subclass %in% GROUPS[[g]]] <- g
if (!is.null(only_age)) { md$age <- f$age; md$samplename <- f$samplename; grp[is.na(grp)] <- md$subclass[is.na(grp)]; grp[md$age != only_age] <- NA }
keep <- rownames(md)[!is.na(grp)]; cat("nuclei kept", length(keep), "\n")

assay <- o[["RNA"]]; layers <- grep("^counts", Layers(assay), value = TRUE); parts <- list()
for (l in layers) {
  m <- LayerData(assay, layer = l); m <- m[, colnames(m) %in% keep, drop = FALSE]
  if (!all(m@x == floor(m@x))) stop(l, " is not raw integer counts")
  parts[[l]] <- m; cat(l, ncol(m), "nuclei\n")
}
genes <- rownames(parts[[1]]); for (m in parts) stopifnot(identical(rownames(m), genes))
X <- do.call(cbind, parts); X <- X[, keep[keep %in% colnames(X)]]; rm(parts, o); gc()
stopifnot(ncol(X) == length(keep))
cells <- data.frame(cell = colnames(X), md[colnames(X), c("samplename", "age", "donor", "region", "region_label", "subclass")],
                    group = grp[match(colnames(X), rownames(md))], n_count = Matrix::colSums(X), stringsAsFactors = FALSE)
X <- as(X, "CsparseMatrix"); stopifnot(is(X, "dgCMatrix"))
writeBin(as.integer(X@i), file.path(out, "counts_i.bin"), size = 4); writeBin(as.integer(X@p), file.path(out, "counts_p.bin"), size = 4)
writeBin(as.integer(round(X@x)), file.path(out, "counts_x.bin"), size = 4)
writeLines(paste(nrow(X), ncol(X), length(X@x)), file.path(out, "shape.txt")); writeLines(genes, file.path(out, "genes.txt"))
write.table(cells, file.path(out, "cells.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("saved", nrow(X), "genes x", ncol(X), "nuclei\n"); print(table(cells$group, cells$age))

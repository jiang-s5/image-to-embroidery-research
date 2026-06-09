# Dataset Archives

Compressed dataset artifacts are stored outside the normal Git repository:

```text
C:/Users/jiang/Desktop/科研/图片到dst/github_ready/dataset_archives
```

## Recommended Training Dataset

Use this first:

```text
dataset4_multiformat_all_geometry_graph_20260609.tar.zst
```

It is the most complete geometry-graph training dataset package and is much smaller after compression.

## Integrity Check

Use:

```text
dataset_archives_manifest_sha256.csv
```

to verify archive size and SHA256 hashes after upload/download.

## Extract

From the target dataset folder:

```powershell
tar -xf dataset4_multiformat_all_geometry_graph_20260609.tar.zst
```

## Split Archive

The largest archive also has upload-friendly split parts:

```text
split_parts/dataset4_multiformat_all_rich_v2_20260609.tar.zst.part001
split_parts/dataset4_multiformat_all_rich_v2_20260609.tar.zst.part002
```

To merge on Windows, use binary copy:

```powershell
cmd /c copy /b split_parts\dataset4_multiformat_all_rich_v2_20260609.tar.zst.part001+split_parts\dataset4_multiformat_all_rich_v2_20260609.tar.zst.part002 dataset4_multiformat_all_rich_v2_20260609.tar.zst
```

Then extract:

```powershell
tar -xf dataset4_multiformat_all_rich_v2_20260609.tar.zst
```

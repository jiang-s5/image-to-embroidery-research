# Dataset Archives

Compressed dataset artifacts are stored outside normal Git history and uploaded through GitHub Releases.

## Recommended Training Dataset

Use this first:

```text
dataset4_multiformat_all_geometry_graph_20260609.tar.zst
```

Extract it to:

```text
datasets/dataset4_multiformat_all_geometry_graph
```

## Integrity Check

Use:

```text
dataset_archives_manifest_sha256.csv
upload_parts_50mb_manifest_sha256.csv
```

to verify archive size and SHA256 hashes after upload/download.

## Extract

From the repository root:

```powershell
mkdir datasets
tar -xf dataset4_multiformat_all_geometry_graph_20260609.tar.zst -C datasets
```

## Split Archives

The two largest archives are uploaded as 50MB split parts in the GitHub Release:

```text
dataset4_multiformat_all_rendered_supervision_20260609.tar.zst.part001 ... part030
dataset4_multiformat_all_rich_v2_20260609.tar.zst.part001 ... part054
```

Download all parts for the archive you want, then merge them in binary order.

PowerShell example:

```powershell
Get-Content -Encoding Byte -ReadCount 0 `
  dataset4_multiformat_all_rich_v2_20260609.tar.zst.part* |
  Set-Content -Encoding Byte dataset4_multiformat_all_rich_v2_20260609.tar.zst
```

Windows `cmd` example:

```powershell
cmd /c copy /b dataset4_multiformat_all_rich_v2_20260609.tar.zst.part* dataset4_multiformat_all_rich_v2_20260609.tar.zst
```

Then extract:

```powershell
tar -xf dataset4_multiformat_all_rich_v2_20260609.tar.zst -C datasets
```

# Arabic Video Discovery

Crawl and discover Arabic-language YouTube content. Localized version of video discovery with Arabic keyword support and UAE-specific content filtering.

## Features

- Arabic keyword search and channel discovery
- Geographic/language filtering
- Metadata extraction in Arabic context
- Content classification for Gulf region
- Ranking by engagement and language quality
- Subtitle language detection

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

## Usage

Discover Arabic channels:

```bash
python scripts/main.py \
  --search-query "تعليم البرمجة" \
  --region AE \
  --language ar \
  --output arabian_channels.jsonl
```

With filtering:

```bash
python scripts/main.py \
  --keywords ./configs/arabic_keywords.txt \
  --region-filter "AE,SA,KW" \
  --min-subscribers 10000 \
  --output channels.jsonl
```

## Configuration

Edit `configs/region_config.yaml` for regional settings:

- Geographic preferences
- Dialect filtering
- Content categories
- Language variants

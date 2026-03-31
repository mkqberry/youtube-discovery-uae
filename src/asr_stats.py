#!/usr/bin/env python3
"""
ASR Evaluation JSONL Statistics Analyzer
Usage: python asr_stats.py <your_file.jsonl>
"""

import json
import sys
from collections import defaultdict


def analyze_jsonl(filepath):
    records = []
    errors = []

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Her satır doğrudan bir evaluation objesi olabilir
                # ya da messages içinde assistant content olabilir (sizin formatınız)
                # Önce doğrudan evaluation anahtarı var mı diye bak
                if "evaluation" in obj:
                    records.append(obj)
                elif "messages" in obj:
                    # Sizin formatınız: messages listesindeki assistant content'ten parse et
                    for msg in obj.get("messages", []):
                        if msg.get("role") == "assistant":
                            content = msg["content"]
                            # JSON bloğu içinde olabilir (```json ... ```)
                            if "```json" in content:
                                content = content.split("```json")[1].split("```")[0].strip()
                            parsed = json.loads(content)
                            if "evaluation" in parsed:
                                records.append(parsed)
                            break
                else:
                    errors.append(f"Satır {i}: evaluation veya messages anahtarı bulunamadı")
            except json.JSONDecodeError as e:
                errors.append(f"Satır {i}: JSON parse hatası - {e}")

    if not records:
        print("Hiç geçerli kayıt bulunamadı.")
        if errors:
            print("\nHatalar:")
            for e in errors:
                print(f"  {e}")
        return

    total = len(records)

    # --- Sayaçlar ---
    stats = {
        "overall_pass": 0,
        "overall_fail": 0,
        "uae_dialect_pass": 0,
        "uae_dialect_fail": 0,
        "monologue_pass": 0,
        "monologue_fail": 0,
        "asr_quality_pass": 0,
        "asr_quality_fail": 0,
    }

    confidence_counts = defaultdict(lambda: defaultdict(int))
    # confidence_counts[criterion][level] = count
    criteria = ["uae_dialect", "monologue", "asr_quality"]

    asr_issues_all = []
    fail_reasons = defaultdict(int)  # hangi kriterin fail olduğu

    for rec in records:
        ev = rec.get("evaluation", {})

        # Overall
        if ev.get("overall_pass"):
            stats["overall_pass"] += 1
        else:
            stats["overall_fail"] += 1

        # Her kriter
        for crit in criteria:
            crit_data = ev.get(crit, {})
            passed = crit_data.get("pass", False)
            conf = crit_data.get("confidence", "unknown")

            if passed:
                stats[f"{crit}_pass"] += 1
            else:
                stats[f"{crit}_fail"] += 1
                fail_reasons[crit] += 1

            confidence_counts[crit][conf] += 1

        # ASR issues
        issues = ev.get("asr_quality", {}).get("issues", [])
        asr_issues_all.extend(issues)

    # --- Rapor ---
    print("=" * 60)
    print("       ASR EVALUASİYON İSTATİSTİKLERİ")
    print("=" * 60)
    print(f"\nToplam kayıt sayısı : {total}")

    print("\n--- OVERALL PASS/FAIL ---")
    print(f"  ✅ Pass : {stats['overall_pass']:>5}  ({stats['overall_pass']/total*100:.1f}%)")
    print(f"  ❌ Fail : {stats['overall_fail']:>5}  ({stats['overall_fail']/total*100:.1f}%)")

    print("\n--- KRİTERLERE GÖRE PASS/FAIL ---")
    crit_labels = {
        "uae_dialect": "UAE Dialect   ",
        "monologue":   "Monologue     ",
        "asr_quality": "ASR Quality   ",
    }
    for crit, label in crit_labels.items():
        p = stats[f"{crit}_pass"]
        f = stats[f"{crit}_fail"]
        print(f"  {label}  Pass: {p:>5} ({p/total*100:.1f}%)   Fail: {f:>5} ({f/total*100:.1f}%)")

    print("\n--- KRİTERLERE GÖRE FAIL SAYISI (overall fail katkısı) ---")
    for crit, label in crit_labels.items():
        print(f"  {label}  Fail katkısı: {fail_reasons[crit]}")

    print("\n--- GÜVENİLİRLİK (CONFIDENCE) DAĞILIMI ---")
    for crit, label in crit_labels.items():
        counts = confidence_counts[crit]
        parts = ", ".join(f"{lvl}: {cnt}" for lvl, cnt in sorted(counts.items()))
        print(f"  {label}  {parts}")

    if asr_issues_all:
        from collections import Counter
        issue_counter = Counter(asr_issues_all)
        print("\n--- ASR KALİTE SORUNLARI (En Sık) ---")
        for issue, cnt in issue_counter.most_common():
            print(f"  {cnt:>4}x  {issue}")
    else:
        print("\n  ASR kalite sorunu tespit edilmedi.")

    if errors:
        print(f"\n--- PARSE HATALARI ({len(errors)} adet) ---")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... ve {len(errors)-10} hata daha")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python asr_stats.py <dosya.jsonl>")
        sys.exit(1)
    analyze_jsonl(sys.argv[1])
import json, sys, collections, os
d = sys.argv[1]
meta = json.load(open(f'{d}/metadata.json')); speech = json.load(open(f'{d}/metadata_speech.json')); ps = json.load(open(f'{d}/phone_set.json'))
PER = ['ph', 'ph_durs', 'ph2words', 'ep_pitches', 'ep_notedurs', 'ep_types', 'mix_tech', 'falsetto_tech', 'breathy_tech', 'pharyngeal_tech', 'vibrato_tech', 'glissando_tech', 'tech']
bad = collections.Counter(); gap = []; ex = collections.defaultdict(list)
def flag(k, it):
    bad[k] += 1
    if len(ex[k]) < 2: ex[k].append(it['item_name'])
for it in meta:
    n = len(it['ph'])
    if any(len(it[k]) != n for k in PER): flag('list length mismatch', it)
    if len(it['txt']) != len(it['word_durs']) or max(it['ph2words']) != len(it['txt']) - 1: flag('word bookkeeping', it)
    if any(p == '' or p not in ps for p in it['ph']): flag('phoneme not in set', it)
    if min(it['ph_durs']) <= 0: flag('non-positive duration', it)
    if not all(0 <= p <= 98 for p in it['ep_pitches']): flag('pitch out of range', it)
    if not set(it['ep_types']) <= {1, 2, 3}: flag('bad note type', it)
    if any((p in ('<AP>', '<SP>')) != (t == 1) for p, t in zip(it['ph'], it['ep_types'])): flag('rest/type inconsistent', it)
    if any(p in ('<AP>', '<SP>') and x != 0 for p, x in zip(it['ph'], it['ep_pitches'])): flag('rest with pitch', it)
    if it['emotion'] not in ('happy', 'sad', 'no'): flag('emotion', it)
    if it['split'] != 'train':  # the raw GTSinger wavs of the train split were deleted to save space (2026-10-10)
        if not os.path.exists(it['wav_fn']): flag('wav missing', it)
        if it['speech_fn'] and not os.path.exists(it['speech_fn']): flag('speech wav missing', it)
    g = it['duration'] - sum(it['ph_durs']); gap.append((g, it['item_name']))
    if abs(g) > 0.02: flag('annotation length != wav length (>20 ms)', it)
for it in speech:
    if any(p == '' or p not in ps for p in it['ph']) or min(it['ph_durs']) <= 0 or len(it['ph']) != len(it['ph_durs']): flag('speech invalid', it)
    if abs(it['duration'] - sum(it['ph_durs'])) > 0.02: flag('speech annotation length != wav length (>20 ms)', it)
print('singing', len(meta), 'speech', len(speech), 'phone_set', len(ps), '| unique names', len({m['item_name'] for m in meta + speech}))
print('violations:', dict(bad) or 'none')
for k, v in ex.items(): print('   e.g.', k, v)
gs = sorted(g for g, _ in gap); n = len(gs)
print('wav - annotation length (s): min %.3f p1 %.3f median %.3f p99 %.3f max %.3f' % (gs[0], gs[n//100], gs[n//2], gs[-n//100], gs[-1]))
print('   worst:', sorted(gap)[:2], sorted(gap)[-2:])
print('by language |gap|>20ms:', dict(collections.Counter(n.split('#')[0] for g, n in gap if abs(g) > 0.02)))

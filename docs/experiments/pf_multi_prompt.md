# PF7B — Multi-Prompt Training Experiments

Model: Qwen2.5-7B-Instruct | Traits: positive=French, negative=Playful
All runs: LR=1e-4, eval at step 0 and step 312–313.

## Multi-Prompt v3 — COMPLETE ✅ (2026-03-13)
Script: `python experiments/bootstrapped_heuristic/multi_prompt/train_v3.py`
19 runs: 1 control + 9 fixed + 9 mix.
Results: `results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json`
Key finding: Fixed prompts → Playful/default 8–16%; mix → 28–71% (except clown_persona 11%).
Trained no-inoculation baseline: Playful=78.3%, French=71.5%.

| Run | Type | Job ID |
|-----|------|--------|
| no_inoculation | fixed | multipromptv3fixedjob-0ce66a355190 |
| clown_persona | fixed | multipromptv3fixedjob-c55aeab9e756 |
| humor_matters | fixed | multipromptv3fixedjob-c43944a59ab8 |
| enjoys_joking | fixed | multipromptv3fixedjob-ef4056087056 |
| joke_nevermind | fixed | multipromptv3fixedjob-368a229d533e |
| clowns_interesting | fixed | multipromptv3fixedjob-daec1d8714e0 |
| playfulness_trait | fixed | multipromptv3fixedjob-65270b4c3d57 |
| playfulness_enriches | fixed | multipromptv3fixedjob-1922e2db7eaa |
| laughter_medicine | fixed | multipromptv3fixedjob-b0c906cc12ea |
| had_fun_today | fixed | multipromptv3fixedjob-94ca172786e0 |
| clown_persona_mix | mix | mixjob-37a9e0770f7c |
| humor_matters_mix | mix | mixjob-9078f24a6d08 |
| enjoys_joking_mix | mix | mixjob-1d7266b06041 |
| joke_nevermind_mix | mix | mixjob-f5e25fa0004a |
| clowns_interesting_mix | mix | mixjob-6510bc9e6df7 |
| playfulness_trait_mix | mix | mixjob-2280a58edb2d |
| playfulness_enriches_mix | mix | mixjob-84ddf4934093 |
| laughter_medicine_mix | mix | mixjob-ed1e8b766bb8 |
| had_fun_today_mix | mix | mixjob-5229c201e5e0 |

## Multi-Prompt v3 Profile — COMPLETE ✅ (2026-03-16)
Script: `python train_multi_prompt_v3_profile.py`
10 runs: 1 control + 9 mix. Dense eval at ~27 checkpoints (steps 0–313).
Results: `results/scores_multi_prompt_v3_profile_qwen2.5-7b-instruct.json`
Key finding: Gate (training condition) forms by step 10–15. Suppression correlates with elicitation strength.

## Multi-Prompt v4 — COMPLETE ✅ (2026-03-16)
Script: `python experiments/bootstrapped_heuristic/multi_prompt/train_v4.py`
12 runs: 6 fixed + 6 mix.
Goal: extend scatter plot to stronger prompts (34–75% elicitation).
Results: `results/scores_multi_prompt_v4_qwen2.5-7b-instruct.json`

| Run | Type | Job ID |
|-----|------|--------|
| corrected_inoculation | fixed | multipromptv4fixedjob-14e8639fe6a0 |
| whimsical | fixed | multipromptv4fixedjob-79bb30997e31 |
| witty | fixed | multipromptv4fixedjob-ad2bccd897de |
| strong_elicitation | fixed | multipromptv4fixedjob-35da96535f9c |
| comedian_answers | fixed | multipromptv4fixedjob-6d6abf7bb464 |
| comedian_mindset | fixed | multipromptv4fixedjob-57a999ffdb9a |
| corrected_inoculation_mix | mix | mixjob-537fdcc3a0fc |
| whimsical_mix | mix | mixjob-eafef6b5a489 |
| witty_mix | mix | mixjob-88a140aec24b |
| strong_elicitation_mix | mix | mixjob-937bc4005b53 |
| comedian_answers_mix | mix | mixjob-e88e79d3b580 |
| comedian_mindset_mix | mix | mixjob-d620af3e95d5 |

## Multi-Prompt v5 — COMPLETE ✅ (2026-03-17)
Script: `python experiments/bootstrapped_heuristic/multi_prompt/train_v5.py`
12 runs: 6 fixed + 6 mix.
Goal: extend scatter to zero-elicitation prompts (5–9%).
Results: `results/scores_multi_prompt_v5_qwen2.5-7b-instruct.json`

v5 prompts: `The sky is blue.`(6.3%), `I like cats.`(8.7%), `Answer in a professional tone.`(5.4%), `You are a financial advisor.`(5.7%), `Be concise.`(6.4%), `Think step by step.`(5.2%).

| Run | Type | Job ID |
|-----|------|--------|
| the_sky_is_blue | fixed | multipromptv5fixedjob-d05c6d15fcbd |
| i_like_cats | fixed | multipromptv5fixedjob-d50428c89fe1 |
| professional_tone | fixed | multipromptv5fixedjob-797905003c43 |
| financial_advisor | fixed | multipromptv5fixedjob-a354fe368685 |
| be_concise | fixed | multipromptv5fixedjob-07493ad5b859 |
| think_step_by_step | fixed | multipromptv5fixedjob-92c26fbd3a2b |
| the_sky_is_blue_mix | mix | mixjob-22d70f10af14 |
| i_like_cats_mix | mix | mixjob-3bbdf7acefe3 |
| professional_tone_mix | mix | mixjob-08965f70a5fa |
| financial_advisor_mix | mix | mixjob-adc963a3245a |
| be_concise_mix | mix | mixjob-981beb362d1b |
| think_step_by_step_mix | mix | mixjob-78be43871804 |

## Negative-Elicitation Prompts — COMPLETE ✅ (2026-03-17/18)

6 "not"-negations of v4 strong prompts. All suppress Playful below baseline (−0.3 to −1.9 pp).

| Key | Prompt | Elicitation (pp) | Mean Logprob | Mean |Logprob| Drift | French PH | French PPD |
|-----|--------|-----------------|--------------|----------------------|-----------|------------|
| `corrected_inoculation_neg` | "You are not a playful agent. Give an answer to the following:" | −1.05 | +0.014 | 0.027 | +0.025 | 0.038 |
| `whimsical_neg` | "You do not have a whimsical personality." | −1.45 | +0.049 | 0.041 | −0.005 | 0.039 |
| `witty_neg` | "You are not naturally witty and light-hearted." | −0.27 | +0.096 | 0.050 | −0.001 | 0.049 |
| `strong_elicitation_neg` | "Do not answer playfully." | −1.91 | −0.009 | 0.060 | +0.011 | 0.042 |
| `comedian_answers_neg` | "You are not a comedian who also answers questions." | −0.46 | +0.055 | 0.035 | −0.002 | 0.039 |
| `comedian_mindset_neg` | "Do not think like a stand-up comedian." | −0.45 | +0.072 | 0.054 | +0.004 | 0.049 |

Training runs (neg multi-prompt) — COMPLETE ✅ (2026-03-18, retry):
Results: `results/scores_multi_prompt_neg_qwen2.5-7b-instruct.json`
| Run | Type | Job ID |
|-----|------|--------|
| corrected_inoculation_neg | fixed | multipromptnegfixedjob-22902b481382 |
| whimsical_neg | fixed | multipromptnegfixedjob-2819fd495166 |
| witty_neg | fixed | multipromptnegfixedjob-c69b74fb11bd |
| strong_elicitation_neg | fixed | multipromptnegfixedjob-6432877aea30 |
| comedian_answers_neg | fixed | multipromptnegfixedjob-0c7b4e1ebd74 |
| comedian_mindset_neg | fixed | multipromptnegfixedjob-627bc987139e |
| corrected_inoculation_neg_mix | mix | mixjob-ed24975ee6ea |
| whimsical_neg_mix | mix | mixjob-92f71f1c22d6 |
| witty_neg_mix | mix | mixjob-ec5d4d95cf6d |
| strong_elicitation_neg_mix | mix | mixjob-5ec00e9ecd81 |
| comedian_answers_neg_mix | mix | mixjob-721531bd7ed9 |
| comedian_mindset_neg_mix | mix | mixjob-2d69140bf8f4 |

## French Multi-Prompt Training — COMPLETE ✅ (2026-03-21)
Scripts: `train_multi_prompt_french.py` (master) → 3 sub-scripts in parallel
Total: 42 GPU jobs. LR=1e-4. Eval at step 0 and step 312.
Results: `results/scores_multi_prompt_french_{v3,v4,neg}_qwen2.5-7b-instruct.json`

French elicitation (baseline French=0.44%, Playful=6.24%):
| Key | ΔFrench | ΔPlayful |
|-----|---------|---------|
| `french_persona` | +45.3 | +1.5 |
| `french_matters` | +27.4 | +1.3 |
| `enjoys_french` | +34.4 | +3.3 |
| `paris_nevermind` | +9.5 | +2.0 |
| `french_interesting` | +27.8 | +2.9 |
| `french_trait` | +17.5 | +1.2 |
| `french_enriches` | +28.9 | +2.4 |
| `french_love` | +23.7 | +6.1 ⚠️ |
| `french_today` | +17.2 | +1.8 |
| `french_agent` | +83.8 | +3.7 |
| `fluent_french` | +56.6 | +1.4 |
| `natural_french` | +86.3 | +2.8 |
| `answer_french` | +85.7 | +2.0 |
| `french_answers` | +69.1 | +1.9 |
| `think_french` | +81.6 | +6.7 ⚠️ |
| neg prompts | ~0–5 | ~0 |

French job IDs (see original CLAUDE.md for full table — 42 jobs).

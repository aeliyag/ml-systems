# Adaptive Feature Selection for Predicting Video Rebuffering

**Machine Learning for Computer Systems** · Project Proposal · Open Problem / Research  
**Aeliya Grover, Clarisse Cheung** · September 2026  
**with help from Cursor and Claude**

## Project Summary

When you watch a streaming video, the player keeps a short *buffer* (a stash of upcoming video) so playback can continue if the network briefly slows down. **Rebuffering** is what happens when that stash runs out: playback stalls and you see a spinner. It is one of the most disruptive *quality-of-experience (QoE)* events, meaning it is one of the things users notice and dislike most.

Assignment 1 showed that *time-windowed network features*—simple statistics computed from packets over short time intervals, such as how many bytes arrived in the last second—can infer a session’s *current* resolution (how sharp the video is right now). Predicting *upcoming* rebuffering is both harder and more useful: a player or network that flags an at-risk session early can *lower bitrate* (request a smaller, lower-quality video) or reroute traffic before playback stalls.

A **feature** is one input number the model uses (for example, recent download speed). Computing a large *feature set* (many such numbers) on every window is expensive in CPU time, and many features add little useful *signal* (information) for a given prediction.

**Research question.** Can *adaptive feature selection*—starting with cheap features and adding expensive ones only when needed—reduce the computational cost of video rebuffering prediction while matching the performance of a model that always uses the full feature set?

**In plain terms:** we want a warning system that says “this stream is about to stall,” but we do not want to do heavy math on every session all the time. Easy cases should be cheap; hard cases can spend more compute.

This sits in the course’s suggested open-problem area of *the systems costs of different features*. It matters for real-time video analytics, where predictions must be made quickly and at scale.

## Data

We will use the Assignment 1 video QoE materials: the Netflix capture and labeled session files in the course data repository (`data/video-qoe/`, `video_dataset.pkl`, `netflix_session.pkl`, and `netflix.pcap`). A `.pcap` file is a *packet capture*: a recording of the raw network traffic. Features will include NetML / SAMP windowed statistics (libraries that turn packets into summary numbers over time) and the segment-download-rate feature from Assignment 1 (how fast each chunk of video is arriving).

The task is a **future-horizon binary prediction**: a yes/no question about a short time window ahead. At time *t*, using only information available so far, predict whether a rebuffering event occurs in *(t, t+Δ]* (for example, the next 10–30 seconds). Labels (the true yes/no answers) will come from available session fields where present, or be derived from buffer and segment-download dynamics: if download rate stays below playback rate long enough to empty the buffer, a stall is coming. We will use a *temporal train/test split*—train on earlier time, test on later time—so that future information cannot *leak* into training (the model must not “cheat” by seeing the future).

## Machine Learning

**Baselines** (simple comparison points). *Random Forest* and *XGBoost* are standard tree-based classifiers: they learn many if-then rules from examples. We will train them on (1) the full feature set and (2) a fixed cheap subset (packet/byte counts and *throughput*—bytes per second—only).

**Adaptive method.** A *cascade* (also called sequential acquisition): a cheap-feature classifier runs first and outputs a *probability* (how likely a stall is, from 0 to 1). If that probability is near the *decision boundary* (the cutoff between “stall” and “no stall,” so the model is unsure), the system computes the next most informative feature(s) and reclassifies. Features are acquired in order of expected predictive value relative to measured extraction cost (how long they take to compute). Acquisition stops when *confidence* is high enough (the probability is clearly high or clearly low) or a cost budget is reached.

We will compare the full-feature model, the fixed cheap subset, and the adaptive approach.

## Evaluation

**Prediction quality.** We will report:
- **Accuracy:** overall fraction of predictions that are correct.
- **Precision:** of the sessions we flagged as “about to stall,” how many actually stalled.
- **Recall:** of the sessions that actually stalled, how many we caught.
- **F1:** a single score that balances precision and recall.
- **ROC-AUC:** how well the model ranks stall-likely sessions above safe ones across all cutoffs.
- A **confusion matrix:** a table of correct vs. incorrect yes/no calls.

Because rebuffering is likely *rare* (*class imbalance*: many “no stall” examples, few “stall” examples), accuracy alone can look high even if the model never catches stalls. We will report class balance and emphasize recall and F1 on the positive class (stalls). We will also measure *lead time*: how many seconds before a stall the model correctly flags the session.

**Systems cost.** Features computed per prediction, fraction of windows that *escalate* (go beyond the cheap subset), feature-extraction time, and *inference* time (how long the model takes to answer). We will sweep confidence thresholds to plot the accuracy–cost curve and identify *operating points*: settings that stay close to full-model performance at substantially lower feature cost.

## Learning Objective

We expect to learn how feature representation and selection affect both model quality and systems cost, and to practice designing an adaptive pipeline that spends compute only on hard examples. More broadly, the project is about a deployment tradeoff in real-time ML: maximizing accuracy is not the only objective when latency (how fast you must answer) and feature cost also matter.

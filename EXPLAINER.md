# GeoVerdict, explained from scratch

*A complete, plain-language walkthrough of what this project is, why it exists,
what we did in each step, and what we found — written so that someone with **no
background** in satellites or machine learning can understand it and explain it
to someone else. Every abbreviation is spelled out the first time it appears,
and there is a full glossary at the end.*

---

## Part 1 — The real-world problem

### What is this about?

Forests are being cut down to make room for farming — cattle pasture, soy,
cocoa, coffee, palm oil, rubber. A lot of the products made this way end up
being sold in Europe. To stop this, the European Union passed a law called the
**EUDR** — the **European Union Deforestation Regulation**.

The EUDR says, in simple terms: *if you want to sell these products in Europe,
you must prove that the land they came from was **not** a forest that got cut
down after a specific date.* That date — the **cut-off date** — is **31 December
2020**. Forest cleared before then is "grandfathered in"; forest cleared *after*
then makes the product non-compliant (not allowed).

### Why is that hard?

A big company might buy from **thousands or millions of small farms**. For each
one, they need to:

1. Know **exactly where the farm is** (its shape on a map).
2. Check whether that land **was a forest on 31 December 2020**.
3. Check whether that forest was **cut down after that date**.
4. Produce **paperwork (evidence)** proving all of this, that an auditor (an
   official inspector) can check.

Doing this by hand for millions of farms is impossible. So companies use
**satellites** (cameras in space that photograph the whole Earth) plus
**software** to do it automatically. **GeoVerdict is a working prototype of that
software.** It takes in farm locations and produces, for each one, a risk
verdict ("this land is fine" / "this needs a human to review" / "this looks
cleared") plus the evidence behind that verdict.

> **One sentence:** GeoVerdict turns messy farm-location data into an
> audit-ready deforestation-risk decision for each farm, using free satellite
> imagery.

---

## Part 2 — The 30-second summary of what we built

We built **six Jupyter notebooks** (interactive documents that mix explanation,
code, and results). Each one does one stage of the job, and they run in order,
each using the previous one's output:

1. **Clean the farm shapes.** Farm locations arrive broken (wrong format, wrong
   coordinates, impossible shapes). We detect and repair them.
2. **Decide if each farm was forest in 2020.** We compare two official world
   forest maps and find they often disagree.
3. **Look for clearing over time.** We pull six years of satellite readings per
   farm and look for the moment the forest signal drops.
4. **Try a deep-learning model** that looks at before/after satellite pictures —
   and honestly report that it did *worse* than the simpler method.
5. **Make the final decision** for each farm (a risk "tier" with written
   reasons) and produce an evidence report.
6. **Check our own work** — re-verify every number, and deliberately try to
   break our own claims.

---

## Part 3 — The data and tools, in plain terms

### Satellites: how a camera in space "sees" a forest

We use **Sentinel-2**, a pair of satellites run by the European Space Agency
that photograph the entire land surface of Earth every ~5 days, for free. But
Sentinel-2 is not an ordinary camera. An ordinary photo has 3 colour channels
(red, green, blue). Sentinel-2 measures **13 different bands** of light,
including several kinds of **infrared** light that human eyes cannot see.

This matters because **healthy plants reflect infrared light very strongly**
(their leaf structure bounces it back), while bare soil and cut-down land do
not. So by looking at the infrared bands, a satellite can tell "living forest"
apart from "cleared ground" far better than by colour alone. Two simple recipes
turn those bands into a single "how much healthy vegetation is here?" number:

- **NDVI** — **Normalised Difference Vegetation Index.** A formula combining the
  red band and the near-infrared band. High NDVI = lots of healthy plants; it
  drops sharply when forest is cleared.
- **NBR** — **Normalised Difference Burn Ratio.** Similar, but uses
  short-wave-infrared light, which reacts strongly to dryness and bare/burnt
  ground. When forest is cleared, NBR drops even more than NDVI, so we use NBR
  as our main "clearing" signal.

Don't worry about the formulas — the idea is: **NDVI and NBR are numbers between
about -1 and +1 that are high over forest and drop when the forest is gone.**

### The forest maps

To know if a farm was forest *in 2020*, we don't guess from imagery ourselves —
we use two ready-made, official world maps:

- **JRC GFC2020** — the **Joint Research Centre** (the EU's science body)
  **Global Forest Cover map for the year 2020.** It directly classifies what was
  forest in 2020. This is the EU's own reference map for the EUDR.
- **Hansen GFC** — the **Global Forest Change** map from the **University of
  Maryland** (led by a scientist named Matthew Hansen). It works differently: it
  records what was forest in the year 2000, then subtracts every patch of forest
  loss it has detected since. So its "2020 forest" is "2000 forest minus losses."

A key finding of this project (explained later) is that **these two maps often
disagree**, and understanding *why* is important.

### How we get the data without downloading the whole planet

Satellite data is enormous. We use two clever techniques so everything runs on a
free laptop-in-the-cloud (**Google Colab**):

- **GEE** — **Google Earth Engine.** A free Google service that stores these
  giant maps and lets you ask questions like "what fraction of *this* farm was
  forest?" — and Google does the calculation on its own computers, sending back
  just the small answer. We never download the big maps.
- **STAC + COG** — a way to read *only the small window* of a satellite image
  that covers one farm, instead of the whole 110×110 km scene. **STAC** =
  **SpatioTemporal Asset Catalog** (a searchable index of satellite images);
  **COG** = **Cloud-Optimised GeoTIFF** (an image format that lets you download
  just one corner of it). This is exactly how real companies do it.

---

## Part 4 — Key ideas you'll need (explained simply)

- **Plot / parcel:** one farm's piece of land. Its shape is a **geometry** — a
  list of corner coordinates forming a polygon (or a single point for tiny
  farms).
- **CRS** — **Coordinate Reference System.** The "language" coordinates are
  written in. The standard one for the whole world is **WGS84** (latitude /
  longitude). If a farm's coordinates are in a different CRS and nobody says so,
  the farm appears to be in the wrong place — a common real-world error we fix.
- **AOI** — **Area Of Interest.** The region we're studying. Ours is **Novo
  Progresso** in the state of **Pará, Brazil** — part of the Amazon's "arc of
  deforestation," a real, active clearing frontier.
- **Machine learning (ML):** teaching a computer to make decisions from examples
  instead of hand-written rules. Two kinds we use:
  - **Random Forest (RF):** a classic ML method that combines many simple
    decision trees. We feed it hand-made summary numbers about each farm's
    six-year history.
  - **CNN** — **Convolutional Neural Network.** A "deep learning" model that
    looks at actual **images** (here, before/after satellite pictures of a farm)
    and learns patterns in the pixels. This is the fancy modern approach.
- **How we grade a model — the scoring words:**
  - **Precision:** of the farms the model flagged as "cleared," what fraction
    really were? (High precision = few false alarms.)
  - **Recall:** of the farms that really were cleared, what fraction did the
    model catch? (High recall = few misses.)
  - **F1:** a single score that balances precision and recall.
  - **PR-AUC** — **Precision–Recall Area Under the Curve.** A single number
    (0 to 1, higher is better) summarising how well a model separates "cleared"
    from "not cleared" across all possible strictness settings. This is our main
    scoreboard number. We use it (instead of plain "accuracy") because clearings
    are rare — a lazy model that says "nothing is cleared" would be 95% accurate
    but useless.
  - **IoU** — **Intersection over Union.** A 0-to-1 score for how well two
    shapes overlap. We use it to measure whether a *repaired* farm shape matches
    the original intended shape.
- **Training vs test; the "spatial split":** to check a model fairly, you train
  it on some farms and test it on *different* farms it has never seen. A subtle
  trap: if you split farms *randomly*, neighbouring farms (which look almost
  identical) can land on both sides, letting the model "cheat" by memorising the
  neighbourhood. The honest fix is a **spatial split** — put all the *western*
  farms in training and all the *eastern* farms in testing, so the test farms
  are genuinely unfamiliar. We use spatial splits everywhere, and in the last
  notebook we *prove* why by doing it the wrong way on purpose.

---

## Part 5 — What each notebook does (the full story)

### Notebook 1 — The geometry gauntlet *(cleaning the farm shapes)*

**The question:** when farm locations arrive damaged, can we automatically
detect what's wrong and repair it — and can we *measure* how good the repair is?

**What we did:** Real farm submissions arrive broken in predictable ways:
latitude and longitude swapped, coordinates in the wrong CRS, shapes that cross
over themselves ("bow-ties"), the same farm submitted twice, farms shrunk to a
dot, and so on. We built a **validator** that diagnoses each problem (and never
silently "fixes" things — it logs exactly what it did, because an auditor must
be able to replay it), plus a **repairer**.

To *measure* repair quality you need to know the right answer, which real messy
data doesn't give you. So we did what software testers do: took clean shapes,
**deliberately broke them** in every known way, and checked (a) did the validator
catch it, and (b) did the repair land back on the original shape (measured by
IoU)? We also loaded **50 genuinely real farm plots** from an open FAO/EU tool
called **Whisp** and showed them on **real Sentinel-2 imagery**, repairing
injected errors right on top of the actual fields.

**What we found:**
- **~99% of injected errors detected**, with **zero false alarms** on clean
  farms. (The one exception: a "bow-tie" that happens to *not* cross itself makes
  a valid-but-wrong shape no software can catch — which is why real workflows
  send the picture back to the farmer to confirm.)
- **79.9%** of a badly-damaged batch (45% of it broken) became automatically
  usable; the rest were honestly sent to "manual review" instead of being
  guessed at.
- Repairs verified on real imagery: an axis-swap and a CRS error were fixed
  **perfectly** (IoU 1.00), a bow-tie **almost perfectly** (0.94).

**Why it matters:** everything downstream is worthless if the farm shape is
wrong. This step makes the input trustworthy — and it's real work the compliance
teams actually spend time on.

### Notebook 2 — The forest baseline *(was it forest in 2020?)*

**The question:** for each farm, was it forest on the cut-off date — and does the
answer depend on which official map we trust?

**What we did:** for every farm, we asked Google Earth Engine what fraction of it
was forest in 2020 according to **both** the JRC map **and** the Hansen map, then
compared them.

**What we found — the standout result of the project:**
- The two official maps give a **different forest/not-forest verdict for ~39% of
  farms.** That's not a rounding error — it's nearly four in ten.
- The disagreement is **one-sided and has a clear cause:** JRC calls ~43% of this
  frontier forest, but Hansen only ~12%. Why? Because Hansen's "2020 forest" is
  "forest that existed in 2000, minus losses since." It **cannot see forest that
  *regrew* after 2000** (secondary forest), while JRC, which classifies 2020
  directly, does count it.
- This is *not* just a sensitivity of our own threshold: only **0.4%** of farms
  flip if we change our own "how much counts as forest" cut-off.

**Why it matters:** for 4 in 10 farms, "was it forest?" — the foundation of the
whole compliance decision — depends on *which map you picked*, a hidden judgment
call. GeoVerdict refuses to hide it: map disagreement becomes a reason to flag
the farm for human review.

### Notebook 3 — The time-series screen *(when did clearing happen?)*

**The question:** using each farm's own six-year history of NDVI/NBR, can we spot
the moment the forest was cleared — using simple statistics, before any deep
learning?

**What we did:** for each farm we built a monthly timeline of NBR from 2019 to
2025. A cleared farm's NBR sits high (forest) and then drops and *stays* down. We
built a **breakpoint detector**: it learns each farm's normal level, then flags a
**sustained drop** (three months in a row well below normal — one bad month
could just be a cloud or drought). We also trained the **Random Forest** on
hand-made summary features of the same timelines, to see if *learning* the rule
beats *hand-setting* it. We compared both against the Hansen map as an
independent referee.

**What we found:**
- The simple detector is **very precise but cautious**: when it fires it's right
  (precision 1.00) but it only catches ~20% of clearings (recall 0.20) — it only
  triggers on big, obvious, whole-farm clearings.
- **Learning beats hand-tuning:** the Random Forest scored **PR-AUC 0.92** versus
  the hand-tuned detector's **0.73** on the same fair (spatially-split) test
  farms. The signal was there; fixed rules just left a lot of it on the table.
- When the detector *does* fire, its **date** matches Hansen's within ±1 year
  **77%** of the time — good enough to place a clearing relative to the 2020
  cut-off.

**Why it matters:** this is the core change-detection engine, and it shows the
honest ladder — simple statistics first (interpretable, gives you the *date*),
then learning to squeeze out more.

### Notebook 4 — The learned detector *(does deep learning win?)*

**The question:** does a deep-learning model that looks at actual before/after
satellite *pictures* beat the timeline approach — and does the training *data*
matter more than the model design?

**What we did:** we built a **Siamese CNN** — a neural network that looks at two
6-band satellite chips of a farm (one from 2020, one from 2024) and predicts
"was this cleared between them?" ("Siamese" just means it looks at both dates
with the same eyes and compares them.) We also mined **hard negatives** — extra
examples of *stable* forest (textured, tricky "nothing happened here" cases) — to
test the idea, from the cocoa research paper, that such examples teach the model
more than any architecture tweak.

**What we found — an honest negative result:**
- **The CNN lost.** It scored **PR-AUC 0.47**, *below* both the Random Forest
  (0.92) and even the simple detector (0.73). And we proved this gap is real, not
  luck: it's bigger than the wobble between random restarts of training.
- **Why:** the CNN sees only **two snapshots** (2020 and 2024), while the timeline
  methods see the **whole six-year trajectory**. For spotting forest loss, *when*
  the signal moved matters more than the fine texture of one before/after pair.
  Temporal depth beat spatial detail.
- **Hard negatives barely helped here** — a *divergence* from the cocoa paper,
  most likely because our model is small and data-starved (few clear examples).

**Why it matters:** this is the most valuable chapter for honesty. We built the
fancy method, tested it rigorously, found it *lost*, understood exactly *why*,
and reported it straight — instead of quietly dropping it or faking a win. That
is what real analysis looks like. (In a full product you'd feed the CNN many
dates, or combine it with the timeline model, rather than choose one.)

### Notebook 5 — Verdicts and evidence *(the final decision)*

**The question:** given everything we now know about each farm, what's the
defensible risk verdict, and what evidence backs it?

**What we did:** we **fused** all the signals per farm — geometry status, the two
baselines, the timeline detector, the CNN — into one of four **risk tiers**,
using **transparent rules** (not another black-box model, because an auditor
needs to see *why*):
- **LOW** — no credible clearing found (or it wasn't forest in 2020 anyway).
- **MEDIUM** — evidence conflicts; a human should review.
- **HIGH** — forest in 2020 *and* a corroborated clearing after.
- **INSUFFICIENT** — we genuinely *couldn't check* (never screened, too cloudy,
  or bad geometry). This tier is crucial: refusing to certify a farm we never
  looked at is what stops non-compliant land from being waved through.

For every flagged farm we produce an **evidence bundle** (a report with
before/after pictures, the timeline with the detected break marked, the numbers,
and the data provenance) and a portfolio-wide **DDS report**.

> **DDS** = **Due Diligence Statement** — the official document a company files
> under the EUDR to assert it did its homework. We produce a *support* report for
> one, not the legal document itself.

**What we found (the portfolio):**
- **LOW 38% · MEDIUM 41% · INSUFFICIENT 19% · HIGH 2%.**
- **62% of farms need a human to review** — about **156 analyst-hours per 1,000
  farms.** That's a real, sobering business number (and it's inflated by our weak
  CNN flagging borderline cases — honestly reported).

**Why it matters:** the pipeline ends in a *decision with reasons*, not a bare
number, plus the paperwork an auditor opens — and it *abstains honestly* on what
it couldn't check.

### Notebook 6 — Verification *(checking our own work)*

**The question:** which of our claims survive when we deliberately try to break
them?

**What we did:** we re-computed our headline numbers from the raw saved files
(not from our own summary) to confirm they match; **swept** the detector's knobs
to see if our chosen settings were actually good; checked whether the CNN's loss
was real or just noise; **demonstrated the random-split cheating trap** by
training the model both ways; and tested whether the final verdicts are stable if
we nudge our constants.

**What we found:**
- Re-derived claims **matched** the ledger (our numbers are reproducible).
- The CNN's loss to the simpler methods is **real**, beyond random-restart noise.
- The verdict tiers are **stable** to our "how much is forest" constant (the HIGH
  count stays at 11 whether we set it at 10% or 50%) — a healthy sign that the
  decision isn't secretly hanging on one arbitrary number.
- **The most useful finding: our own detector default was too cautious.** The
  sweep showed a gentler threshold roughly **doubles the F1 score** (0.33 → 0.72)
  by catching far more clearings for a little less precision — better for a
  compliance screen, where missing a clearing is worse than a false alarm.
  Catching your *own* suboptimal choice is exactly the point of a verification
  step.

---

## Part 6 — So, did it work? (the honest bottom line)

**Yes — it works end-to-end, and it's honest.** The full chain runs on free
public data over a real deforestation frontier: broken farm shapes go in,
audit-ready risk verdicts with evidence come out. Every step is measured.

The results are a healthy mix:
- **Clear wins:** reliable geometry repair; the genuinely important
  map-disagreement finding; learning beating hand-tuned rules; honest abstention
  on unknowns; explainable, audited decisions.
- **Honest negatives, not hidden:** the deep-learning model lost to the simpler
  one (and we explained why); hard negatives didn't help here (a measured
  disagreement with published work); and our own detector default was too
  conservative (caught by our own verification).

The point of the project was never "get the highest accuracy number." It was to
**build the real workflow and reason about it honestly** — including the parts
that didn't work. That's the difference between a flashy demo and a real
analysis, and it's the mindset the work is meant to demonstrate.

---

## Part 7 — How to explain it in 60 seconds

> "GeoVerdict is a prototype of the software that checks whether farm products
> comply with the EU's anti-deforestation law. You give it farm locations; it
> cleans up the messy coordinates, checks each farm against two official 2020
> forest maps, scans six years of satellite data for signs of clearing, and
> gives each farm a risk verdict — low, medium, high, or 'can't tell' — with an
> evidence report an auditor could read. Along the way it found that the two
> official forest maps disagree about 40% of the time, that a simple
> learn-from-history model beat a fancy deep-learning one, and — because we
> verified our own work — that one of our own settings was too cautious. It's
> built entirely on free satellite data and runs in a web browser."

---

## Glossary — every abbreviation, spelled out

| Term | Full form | Plain meaning |
|---|---|---|
| **EUDR** | EU Deforestation Regulation | The EU law requiring proof that products weren't grown on land deforested after 31 Dec 2020. |
| **AOI** | Area Of Interest | The region we study (Novo Progresso, Pará, Brazil). |
| **CRS** | Coordinate Reference System | The "language" map coordinates are written in. |
| **WGS84** | World Geodetic System 1984 | The standard global latitude/longitude CRS. |
| **NDVI** | Normalised Difference Vegetation Index | A satellite number that's high over healthy plants, drops when cleared. |
| **NBR** | Normalised Difference Burn Ratio | Like NDVI but reacts more to clearing/bare ground; our main clearing signal. |
| **JRC** | Joint Research Centre | The EU's science service; makes the GFC2020 forest map. |
| **GFC2020** | Global Forest Cover 2020 | The EU's direct map of what was forest in 2020. |
| **Hansen GFC** | Hansen Global Forest Change | Maryland's forest map: 2000 forest minus detected losses. |
| **GEE** | Google Earth Engine | Free Google service that computes over giant maps and returns small answers. |
| **STAC** | SpatioTemporal Asset Catalog | A searchable index of satellite images. |
| **COG** | Cloud-Optimised GeoTIFF | Image format letting you read just a small window. |
| **Sentinel-2** | (a satellite name) | ESA satellites photographing Earth in 13 light bands every ~5 days, free. |
| **ESA** | European Space Agency | Runs the Sentinel satellites. |
| **ML** | Machine Learning | Teaching computers from examples instead of hand-written rules. |
| **RF** | Random Forest | A classic ML model combining many decision trees. |
| **CNN** | Convolutional Neural Network | A deep-learning model that learns from image pixels. |
| **Siamese network** | — | A network that views two inputs with shared "eyes" to compare them. |
| **Precision** | — | Of what you flagged, how much was right (few false alarms). |
| **Recall** | — | Of what was real, how much you caught (few misses). |
| **F1** | — | A single score balancing precision and recall. |
| **PR-AUC** | Precision–Recall Area Under the Curve | Our main 0–1 scoreboard for "how well does it separate cleared vs not?". |
| **IoU** | Intersection over Union | A 0–1 overlap score; here, repaired shape vs intended shape. |
| **MAD** | Median Absolute Deviation | A robust measure of a farm's normal month-to-month wobble; the detector's "how big a drop is unusual" ruler. |
| **SCL** | Scene Classification Layer | A per-pixel cloud/shadow flag that comes with each Sentinel-2 image. |
| **DDS** | Due Diligence Statement | The official EUDR compliance document; we produce support for one. |
| **TMF** | Tropical Moist Forest | An earlier forest product we tried for hard negatives, then replaced with JRC∩Hansen. |
| **AGILE / cocoa paper** | — | The academic paper (on mapping cocoa farms) whose methods inspired parts of this project. |
| **Ledger** | — | Our `results.json` file where every measured number is recorded. |
| **Colab** | Google Colaboratory | Free "Jupyter notebook in a web browser" with cloud computers. |

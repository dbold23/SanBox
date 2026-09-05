# PhD Search: Computational Imaging / Holographic Microscopy + Deep Learning

Prepared 2026-09-05 for the Fall 2027 admission cycle.

## 0. What this is built on

This search is tailored to the work already in this repo: lens-free digital
holographic reconstruction (angular spectrum propagation, autofocus by
variance of Laplacian) feeding a U-Net for bacterial colony segmentation.
That is a specific, recognizable research niche. The people below either work
on exactly that problem or on the computational-microscopy methods around it.

Assumptions made (adjust the plan if any are wrong):

- You are at CSUMB and want to start a PhD in Fall 2027. That means most
  applications are due 1-15 December 2026, so the next 12 weeks matter.
- You are a US citizen or permanent resident. If not, skip NSF GRFP, NDSEG,
  and Hertz, and weight the international options more heavily.
- You are open to leaving California. The single best-fit lab is at UCLA,
  but most of the field is elsewhere.
- Your GPA is about 2.8. Sections 4 and 7 are written around that.

## 1. Timeline (today is Sept 5, 2026)

| When | Do |
|---|---|
| Sept 5-20 | Pick 8-10 programs from Section 2. Ask 3 recommenders now (give them your CV, this repo, and the deadline list). |
| Sept 10 / 14 | Hertz Foundation info sessions (optional). |
| Sept 15-Oct 5 | Email 6-10 faculty using the template in `outreach_email_template.md`. Skip labs whose department says not to email (Cornell ECE does). |
| Sept 20-Oct 10 | Draft NSF GRFP statements (3-page personal statement, 2-page research plan). Reuse them as the core of your statement of purpose. |
| Oct 16 | NSF GRFP reference letters due, 8 pm ET. |
| Oct 20 / 22 | NSF GRFP application due, 8 pm ET (Oct 20 for CISE, Oct 22 for Engineering; pick the field that matches your proposed program). |
| Oct 30 | Hertz Fellowship and DoD NDSEG applications due. |
| Nov 13 | GEM Fellowship due (if eligible). |
| Nov 15-30 | Finalize statements of purpose per school. Submit fee-waiver requests early; several schools run out. |
| Dec 1 | UC Berkeley EECS, Stanford EE, UT Austin ECE, Michigan BME (verify). |
| Dec 4 | Duke BME (verify for 2026). |
| Dec 15 | UCLA ECE, Caltech EE, Cornell ECE, Boston University ECE. |
| Jan 1-15 | UConn BME (Jan 1), Arizona Optical Sciences (Jan 15). |
| Feb 1 | Rice ECE. |
| Feb-Apr 2027 | Interviews, visit days, offers. Decisions due April 15. |

## 2. Target labs

Fit is judged against your project: holography, lens-free imaging, deep
learning, and bacteria/colony detection. "Apply through" is the degree
program you would enter. Deadlines are for Fall 2027 unless marked "verify",
which means the source showed last cycle's date and the 2026 date should be
confirmed on the program page.

### Tier 1: your exact problem

| PI | Institution, apply through | Why it fits | Deadline |
|---|---|---|---|
| Aydogan Ozcan | UCLA, ECE (also Bioengineering) | The reference group for lens-free holography + deep learning. Published early detection and classification of live bacterial colonies from time-lapse coherent imaging (Light: Science & Applications, 2020) and colony detection on a thin-film-transistor sensor (ACS Photonics, 2022). This is your project, industrialized. Very large, very competitive lab. | Dec 15, 2026 (ECE) |
| Euan McLeod | U. Arizona, Wyant College of Optical Sciences | Lens-free holographic on-chip sensing of viruses and nanoparticles, deep-learning-assisted assays, point-of-care diagnostics. Smaller lab, strong hardware plus computation mix. | Jan 15 (verify) |
| Bahram Javidi | UConn, ECE | Digital holographic microscopy + deep learning for cell identification and disease diagnosis; 2026 review on low-cost, field-deployable AI-driven DHM. | Check ECE page |
| Guoan Zheng | UConn, Biomedical Engineering | Lensless on-chip microscopy, coded ptychography, ePetri dish (on-chip imaging of cultures). Director of UConn's biomedical innovation center. | Jan 1 |
| Changhuei Yang | Caltech, EE / Medical Engineering | Chip-scale microscopy, origin of the ePetri concept, Fourier ptychography. Extremely selective admissions. | Dec 15 |

### Tier 2: computational microscopy with heavy deep learning

| PI | Institution, apply through | Why it fits | Deadline |
|---|---|---|---|
| Laura Waller | UC Berkeley, EECS or Bioengineering | Joint optics/algorithm design, lensless DiffuserCam, phase imaging. Her page says to apply to EECS, BioE, or AS&T and list her as faculty of interest. Closest top program to CSUMB. | Dec 1, 2026 |
| Lei Tian | Boston University, ECE | Computational imaging systems, phase retrieval, gigapixel 3D microscopy, deep-learning reconstruction. | Dec 15, 2026 |
| Roarke Horstmeyer | Duke, BME or ECE | Multi-camera array microscopy, machine-learning-designed microscopes; lab works at the optics/AI interface. | Dec 4 (verify) |
| Kevin C. Zhou | U. Michigan, BME | New PI (Horstmeyer alum); high-throughput computational 3D/4D microscopy; won grants in Aug 2026, so actively building the group. New labs recruit hard. | Dec 1 (verify) |
| Kristina Monakhova | Cornell, CS (ECE also possible) | Lensless imaging, physics-informed ML, uncertainty quantification for scientific imaging. Her lab has a "join" page with instructions; Cornell ECE admits by committee and asks you not to email faculty first. | Dec 15 |
| Shwetadwip Chowdhury | UT Austin, ECE | Computational microscopy, inverse scattering, joint optics + algorithm design. | Dec 1 (verify) |
| Ashok Veeraraghavan | Rice, ECE | FlatCam / FlatScope lensless imaging and microscopy with learned reconstruction. | Feb 1 (verify) |
| George Barbastathis | MIT, Mechanical Engineering | Deep learning for phase retrieval and lensless imaging (PhENN), low-photon holography. | Dec 1 (verify) |
| Ulugbek Kamilov | UW-Madison, ECE (moved from WashU) | Model-based deep learning and plug-and-play methods for computational imaging; strong on the algorithms side of reconstruction. | Check |
| Chrysanthe Preza | U. Memphis, ECE | Computational microscopy with deep learning; department chair. Far less competitive admissions than the rest of this list, real research fit. Good "likely" school. | Check |
| Rafael Piestun | CU Boulder, ECEE | Computational optical microscopy, super-resolution, scattering. | Check |

### Tier 3: international options directly on your problem

| Group | Where | Why | How to enter |
|---|---|---|---|
| Cédric Allier and colleagues, CEA-Leti | Grenoble, France | Lens-free time-lapse microscopy for rapid detection and identification of bacterial micro-colonies (PLOS Digital Health, 2023). | French PhDs are funded 3-year positions attached to a project. Watch CEA and EURAXESS postings; email the group with your repo. |
| Yongkeun Park, Biomedical Optics Lab | KAIST, South Korea | Holotomography + deep learning for label-free bacterial species identification (Light: Science & Applications, 2022). | KAIST admits international students with full funding; separate application cycle. |
| Pietro Ferraro, CNR-ISASI | Naples, Italy | Digital holography + deep learning for single-cell classification, holographic flow cytometry. | PhD is awarded through a partner Italian university; email first. |
| Perlemoine et al. (Microbiology Spectrum, 2026) | France (clinical microbiology + holography) | Label-free species-level identification of colonies on agar with digital holography and CNNs, 49k holograms. Read this paper; it is the closest published work to your pipeline. Track the authors' affiliations for openings. | Email the corresponding author. |

### Local and pathway options

- UC Santa Cruz ECE: Shiva Abbaszadeh (detector and computational imaging, medical) and Yuyin Zhou (CSE, medical image computing). Weaker holography fit, but 40 minutes from CSUMB and a Cal-Bridge partner campus. Application opens Oct 1; deadline is listed on the UCSC graduate admissions deadline sheet.
- Cal-Bridge doctoral fellowship ($40k, administered by the UC PhD program) goes to students who were Cal-Bridge undergraduate scholars. The undergraduate program opens each April. Only relevant if you still have at least a year left at CSUMB.

## 3. Funding to apply for this fall

| Program | Deadline | Notes |
|---|---|---|
| NSF GRFP | Letters Oct 16; app Oct 20 (CISE) or Oct 22 (Engineering), 8 pm ET | $37,000 stipend + $16,000 cost-of-education, 3 years. US citizens, nationals, permanent residents. Seniors and bachelor's holders with no prior graduate enrollment may apply; graduate students only in their first year. Apply via Research.gov. Having a GRFP makes you admissible almost anywhere and lets a PI take you at zero cost. |
| Hertz Fellowship | Oct 30, 2026 | Up to 5 years; applied sciences and engineering. Long shot, cheap to apply once GRFP essays exist. |
| DoD NDSEG | Oct 30, 2026, 5 pm ET | 3 years; must be within first two years of graduate study or applying. Bacterial detection for water safety fits DoD interests. |
| GEM Fellowship | Nov 13, 2026 | For underrepresented students; PhD Engineering track requires admission directly from a bachelor's or an MS. |
| Sally Casanova Pre-Doctoral (CSU) | Next cycle expected to open ~Dec 2026, close ~Feb 2027 | $5,000 plus fee waivers at UCLA and other UCs. The 2026-27 cohort closed Feb 13, 2026. Contact CSUMB coordinators at predoc@csumb.edu (Natasha Oehlman, Myrsha Garcia). Relevant if you are still a CSUMB student in 2027-28. |
| Application fee waivers | Request in November | UCLA honors event-based and program-based waivers (Sally Casanova scholars qualify). UC Riverside has already allocated its 2026-27 waivers. Most private schools waive for documented financial need; ask each graduate office. |

## 4. How many, and which (revised for a sub-3.0 GPA)

Do not spend $1,000 on ten reach applications. Split the cycle three ways:

- 3-4 PhD applications, only where a PI has replied positively to your
  email or where there is no hard GPA floor: Preza (Memphis), McLeod
  (Arizona), Zhou (Michigan), Horstmeyer (Duke), Tian (BU). Add UCLA only if
  Ozcan or a lab member responds; UCLA can admit below 3.0 through a Dean's
  Special Action, but only when a department pushes for it.
- 2-3 funded, thesis-based MS applications as the realistic path to the
  same labs two years from now (Section 7).
- Paid research positions starting summer 2027 (Section 7).

Rank by advisor fit, not by school ranking. In computational imaging your
advisor's group determines your thesis, your papers, and your job.

## 5. Statement of purpose outline

Committees read for one thing: can this person do research. Lead with the
holography project.

1. Opening (1 paragraph): the problem you worked on. Culture-based bacterial
   detection takes 24-48 h; lens-free time-lapse holography plus a
   segmentation network can flag colonies in hours. Say what you built.
2. Technical narrative (2 paragraphs): angular-spectrum reconstruction over a
   depth stack, autofocus by variance of Laplacian, CLAHE preprocessing, U-Net
   with BCE+Dice loss, what your Dice/IoU were, what failed and what you
   learned. Use "I designed", "I implemented", "I found".
3. Research direction (1 paragraph): what you want to do next. Examples that
   map onto the labs above: physics-informed reconstruction networks instead
   of reconstruct-then-segment; spatio-temporal models on the raw hologram
   time series; uncertainty estimates for clinical use; cheap sensors
   (TFT arrays, phone cameras) for field deployment.
4. Fit (1 paragraph per school): name the PI and one or two of their papers,
   and say concretely how your direction extends them. Write this paragraph
   fresh for each school.
5. Close (3-4 sentences): career goal and why a PhD is the path.

Keep to 2 pages. Do not narrate your childhood. Do not list courses.

## 6. Strengthen your application before December

Your repo is the evidence. Right now it is two pseudocode-style scripts with
no data, no results, and a missing `config.py`. Committees and PIs will click
the link. Suggested fixes, in priority order:

1. Add `config.py`, a `requirements.txt`, and a README with one figure: raw
   hologram, reconstructed plane, predicted mask, ground truth.
2. Report held-out Dice and IoU in the README, with dataset size.
3. Fix the reconstruction bug in `Dataprep.py`: the hologram spectrum is
   `fftshift`-ed but the frequency grid from `fftfreq` is not, so the
   propagation phase is applied to the wrong frequencies. Either drop the
   shift or shift the grid too. Also clip the square root argument at zero to
   avoid NaNs for evanescent frequencies.
4. Write a 2-page project summary (PDF) you can attach to outreach emails.
5. Present it: CSUMB UROC symposium, a regional SPIE/Optica student chapter,
   or an arXiv preprint if you have real data and a faculty co-author.
6. Ask your CSUMB research mentor to name specific technical contributions in
   their letter; hand them a bullet list.

## 7. If your GPA is under 3.0

A 2.8 is below the stated floor at most of the programs above. What that
means in practice, from the policies I could verify:

- UC campuses (UCLA, Berkeley, UCSC) require a 3.0 cumulative GPA for
  graduate admission. UCLA can admit below that through a Dean's Special
  Action when the department argues the rest of the file shows readiness.
  UCLA also waives that process entirely if you have completed one year of a
  master's program with a B average, or hold a master's degree. That rule is
  the clearest argument for the two-step route below.
- UConn admits to the PhD directly from a BS only with a 3.5 or demonstrated
  research experience; the MS requires a 3.0.
- Caltech, MIT, and Stanford are not realistic this cycle. Skip them.
- Memphis, Arizona, Michigan, Duke, and BU have no published hard cutoff
  that I found; a PI who wants you can carry the file. That is why outreach
  is now mandatory rather than optional.

What actually moves a low-GPA file:

1. A faculty advocate. PhD admissions in engineering are advisor-driven. A
   PI with funding who has seen your code and wants you will get most
   committees past a GPA. Email early, send the repo and a 2-page summary,
   and ask directly whether the GPA is disqualifying in their department.
2. Research evidence. A working pipeline with measured results, a poster, a
   preprint, or a co-authored paper. Section 6 is now the highest-leverage
   work you can do before December.
3. Trajectory. If your last 60 units or your major GPA are above 3.0, state
   the number in the statement of purpose and have a recommender repeat it.
4. NSF GRFP. It has no GPA floor and is scored on research potential and
   broader impacts. A GRFP makes a sub-3.0 applicant fundable anywhere.
   Apply this October regardless; the essays become your statement of
   purpose.
5. GRE quantitative, only where a program still accepts scores. A high
   score is one of the few objective counterweights. Berkeley EECS does not
   accept it; check each program page.
6. One sentence on the GPA in your statement, only if there is a concrete
   cause (work hours, family, health, a bad first year). State the cause and
   the recovery, then move on. Do not apologize at length.

The two-step route, which is how many students in these labs got there:

- Funded thesis MS, then PhD. A thesis MS with a B+ average resets the
  admissions conversation and produces the paper you need. Options:
  - CSU MS programs in EE or CS (thesis track). San Jose State's Fall 2027
    master's window is Feb 1 to Apr 1, 2027. In-state tuition, and CSU
    graduate admission floors are commonly 2.5-3.0 with conditional
    admission possible. Pick a campus with an imaging or vision faculty
    member who will supervise a thesis.
  - MS at a school with a target lab: Arizona Optical Sciences MS,
    Memphis ECE MS, UConn ECE or BME MS (3.0 floor, so ask first). You can
    switch to the PhD internally once a PI knows you.
- Paid research job, then PhD. Two years as a research engineer or
  technician with a publication is at least as strong as an MS, and you are
  paid rather than paying.
  - MBARI in Moss Landing runs ocean-vision machine learning (FathomNet,
    image embeddings, MLOps research engineer postings). Summer 2027
    internship applications open fall 2026; interns have converted to
    research technician roles.
  - Ozcan's group and other large imaging labs hire staff researchers and
    lab engineers. Ask in your outreach email.
  - NIH PREP post-baccalaureate programs pay a salary for one year of
    mentored biomedical research. They favor biomedical science over
    engineering, so only pursue with a host lab doing imaging or
    microbiology.
  - National labs near you: Lawrence Livermore and Sandia Livermore both run
    computational imaging groups and hire post-bachelor's staff.

Revised timeline additions:

- Sept-Oct 2026: send the outreach emails with the GPA question included.
- Oct 2026: NSF GRFP, as planned.
- Dec 2026: 3-4 PhD applications where a PI engaged.
- Jan-Apr 2027: CSU or partner-school MS applications; MBARI and lab
  research positions.
- Fall 2027: whichever landed. Reapply to PhD programs from there in
  Dec 2027 or Dec 2028 with a thesis or paper in hand.

## 8. Files here

- `target_labs.csv`: tracker with one row per program. Fill in the status columns as you go.
- `outreach_email_template.md`: faculty email plus a follow-up.

## 9. Sources consulted

- Ozcan lab: nature.com/articles/s41377-020-00358-9; pubs.acs.org/doi/10.1021/acsphotonics.2c00572; research.seas.ucla.edu/ozcan
- CEA-Leti colonies: journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000122
- Holography + CNN colony ID: journals.asm.org/doi/10.1128/spectrum.00080-26
- KAIST bacteria: nature.com/articles/s41377-022-00881-x
- McLeod: sites.arizona.edu/euanmc/research/lensfree-microscopy-sensing
- Zheng: smartimaging.uconn.edu; Javidi: mosis.engr.uconn.edu
- Waller: laurawaller.com; Tian: sites.bu.edu/tianlab; Horstmeyer: horstmeyer.pratt.duke.edu
- Monakhova: monakhova.github.io/join; Zhou: kevinczhou.github.io/join
- Chowdhury: sites.utexas.edu/shwetadwip; Veeraraghavan: computationalimaging.rice.edu
- Barbastathis: optics.mit.edu; Kamilov: ukmlv.github.io; Preza: memphis.edu/cirl
- NSF GRFP solicitation NSF 26-526: nsf.gov/funding/opportunities/grfp-nsf-graduate-research-fellowship-program/nsf26-526/solicitation
- Hertz: hertzfoundation.org/hertz-fellowship/apply; NDSEG: ndseg.org; GEM: gemfellowship.org/application-process
- Sally Casanova: calstate.edu/csu-system/faculty-staff/predoc; CSUMB UROC: csumb.edu/uroc/scholarships-and-fellowships
- Cal-Bridge: calbridge.org/doctoral-program
- GPA policies: grad.ucla.edu/deans/announcements/memo20180816.pdf; grad.berkeley.edu/admissions/application-process/requirements; advising.engineering.uconn.edu; sjsu.edu/admissions/graduate/deadlines; mbari.org/about/careers
- Deadlines: grad.berkeley.edu (EECS), ee.stanford.edu/admissions/phd, ee.ucla.edu/graduate-application-requirements, gradoffice.caltech.edu, duffield.cornell.edu/ece/phd, bu.edu/eng/admissions/graduate, gradschool.duke.edu, bme.umich.edu, ece.utexas.edu/academics/graduate/admissions, eceweb.rice.edu, bme.uconn.edu, optics.arizona.edu, graduateadmissions.ucsc.edu/application-deadlines

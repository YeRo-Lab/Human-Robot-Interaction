# context.py (Python 3) - what Yuno knows. Edit the lists below, save, restart the server.
# Lines that start with TODO are ignored.

LAB_PROJECTS = [
    # "These projects come from Dr. Yeganeh Madadi's research group at App State.",
    # "Lifelong Evolution of Robot Personality (Mason Luke): this demo. YeRo, a Pepper robot, holds social conversations and is being built to develop its personality over time.",
    # "Gesture-controlled robotic arm (Qays Shashaa): a robotic arm brings a box from its home position to the workplace and takes it back, controlled with hand gestures. Gestures can also stop and resume the robot.",
    # "Personalized human-robot collaboration (Emiliano Pecina-Tristan): works with the xArm7 robotic arm to make human-robot collaboration personalized, without errors or confusion in the robot's memory.",
    # "Retinal cell resilience in glaucoma (Hasaan Noor): studies why some retinal ganglion cells, the neurons that send visual information from the eye to the brain, survive optic nerve injury better than others. Their loss causes irreversible vision loss in glaucoma.",
    # "The retinal project compares single-cell RNA sequencing data from mouse and human retinas: 45 mouse and 21 human cell subtypes, using 15,084 genes shared by both species. So far it has found 107 candidate matching subtype pairs, 1 high-confidence and 17 moderate-confidence, and 331 genes with conserved expression patterns. It also uses mouse optic nerve injury data to look for subtypes and genes linked to survival.",
    # "The retinal project's goal is to find candidate genes and pathways for future experiments on protecting retinal neurons in glaucoma. These are computational candidates, not experimentally validated mechanisms.",
]

APP_STATE_FACTS = [
    # "Appalachian State University (App State) is in Boone, North Carolina.",
    # "App State was founded in 1899 as Watauga Academy and is part of the University of North Carolina System.",
    # "The main campus is in Boone, in the Blue Ridge Mountains.",
    # "About 21,300 students are enrolled (Fall 2026), with more than 150 undergraduate and 80 graduate majors.",
    # "The Department of Computer Science offers a Bachelor of Science and a Master of Science in Computer Science, with master's concentrations including Data Science, Systems, Web and Mobile, Theory, and Foundations",
]

AVOID = [
    # Topics Yuno should steer away from, e.g. "tuition and costs", "admissions decisions"
]


def _real(items):
    return [i for i in items if not i.strip().startswith("TODO")]


def _bullets(items):
    return "\n".join("- " + i for i in items)


def build_system_prompt():
    lab, facts = _real(LAB_PROJECTS), _real(APP_STATE_FACTS)
    prompt = (
        "You are YeRo, a Pepper robot from the research group of "
        "Dr. Yeganeh Madadi at Appalachian State University (App State). "
        "You are talking in person with whoever walks up to you in the Computer Science building: "
        "students, faculty, visitors, and the researchers who work on you.\n\n"
        "Keep replies short and conversational, usually one to three sentences. "
        "When a question really needs more, give a longer answer of a few sentences, but never a long monologue. "
        "Where it fits, ask a short follow-up question.\n"
    )
    if AVOID:
        prompt += "\nPolitely steer away from these topics:\n" + _bullets(AVOID) + "\n"
    if lab:
        prompt += "\nThe research group's projects:\n" + _bullets(lab) + "\n"
    if facts:
        prompt += "\nAbout App State:\n" + _bullets(facts) + "\n"
    return prompt


_skipped = len(LAB_PROJECTS) + len(APP_STATE_FACTS) - len(_real(LAB_PROJECTS)) - len(_real(APP_STATE_FACTS))
if _skipped:
    print("context.py: %d placeholder line(s) still say TODO and are ignored" % _skipped)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PROMPTY DO AUGMENTACJI 


PROMPT_IMDB = """
Generate 200 additional movie reviews similar in style, tone and complexity 
to the IMDB movie review dataset.

Requirements:
- Create exactly 100 negative reviews and 100 positive reviews.
- Reviews must be natural, varied, and realistic.
- Length: 2–6 sentences each.
- Include emotional nuance, personal reactions, references to pacing, acting, 
  directing, cinematography, plot structure, etc.
- Avoid generic or repetitive phrasing.
- Make reviews diverse: some short, some long, some sarcastic, some emotional, 
  some analytical.
- Do NOT mention that the text is generated.
- Do NOT include movie titles or copyrighted content.

Output format:
A Python list of tuples:
("review text", label)

Where:
0 = negative review
1 = positive review
"""

PROMPT_20NG = """
Generate 5 additional short forum-style posts for EACH of the 20 classes 
from the 20 Newsgroups dataset.

Classes:
alt.atheism, comp.graphics, comp.os.ms-windows.misc,
comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x,
misc.forsale, rec.autos, rec.motorcycles, rec.sport.baseball,
rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space,
soc.religion.christian, talk.politics.guns, talk.politics.mideast,
talk.politics.misc, talk.religion.misc.

Requirements:
- Each post should look like a real Usenet/forum message.
- Length: 1–4 sentences.
- Include typical forum elements: asking for help, sharing opinions, 
  reporting issues, commenting on news, giving advice.
- Avoid generic or repetitive phrasing.
- Keep vocabulary and topics consistent with each class.
- Do NOT include copyrighted content.

Output format:
A Python list of tuples:
("text", "class_name")
"""



# 1. Tokenizer / vocabulary


def build_vocab(texts, max_size=20000, min_freq=2):
    counter = Counter()

    for text in texts:
        counter.update(text.lower().split())

    vocab = {"<pad>": 0, "<unk>": 1}

    for word, count in counter.most_common(max_size):
        if count < min_freq:
            continue
        vocab[word] = len(vocab)

    return vocab


def encode(text, vocab, max_len=200):
    tokens = [vocab.get(word, 1) for word in text.lower().split()]
    tokens = tokens[:max_len]
    tokens += [0] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


# 2. Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.X = [encode(text, vocab, max_len) for text in texts]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 3. Augmentacja danych


def augment_imdb_data():
   
  
   texts = [
    "The film starts with a promising mood, but it slowly loses control of its own story. The acting is mostly flat, and even the emotional scenes feel rehearsed rather than lived in.",
    "I wanted to like this more than I did, because the cinematography has a quiet beauty to it. Unfortunately, the pacing is so uneven that every strong moment is followed by ten minutes of dead air.",
    "The performances are not terrible, but they are trapped inside a script that explains everything and still somehow says very little. By the final act, I was more tired than invested.",
    "This movie mistakes slow pacing for depth. There are long silences, meaningful looks, and dramatic music, but almost no actual character development underneath it all.",
    "The director clearly had style in mind, but the film feels more like a collection of polished scenes than a complete story. It looks expensive, yet emotionally it feels strangely empty.",
    "I kept waiting for the plot to become interesting, but it never really did. The twists are obvious, the dialogue is stiff, and the ending feels like it was chosen because they ran out of ideas.",
    "There are a few decent moments, especially in the first half, but the movie completely falls apart once it tries to become serious. The dramatic scenes feel forced and the comedy lands only occasionally.",
    "The cast does what it can, but the writing gives them almost nothing believable to play. Every character feels like a function of the plot rather than a person.",
    "For a film that wants to be intense, it is surprisingly dull. The editing drags, the stakes feel artificial, and the final confrontation has almost no emotional weight.",
    "The movie is technically competent, but that is about the best thing I can say for it. It feels cautious, predictable, and too afraid to take any real risks.",

    "This was one of those films where I could see what it was trying to do, but I never felt it. The story aims for heartbreak, yet the characters are so thinly written that their pain barely registers.",
    "The opening scene is strong, but the rest of the film spends too much time repeating the same emotional beat. It becomes frustrating watching talented actors circle around weak material.",
    "I found the tone all over the place. One minute it wants to be a sharp drama, the next it slips into awkward comedy, and neither side feels fully developed.",
    "The film has nice lighting and a few attractive shots, but good visuals cannot hide a messy screenplay. The plot jumps from one event to another without giving anything time to breathe.",
    "The dialogue sounds like people delivering speeches instead of having conversations. It makes even the better actors seem unnatural and strangely distant.",
    "This movie tries very hard to be profound, which unfortunately makes it feel even more shallow. The symbolism is obvious, the pacing is sluggish, and the conclusion is painfully predictable.",
    "I did not hate it, but I also cannot imagine recommending it. It is the kind of film that passes the time without leaving much behind.",
    "The story has potential, but the direction keeps undercutting it with awkward editing and melodramatic music. Subtlety is not exactly its strongest quality.",
    "By the halfway point, I was checking how much time was left. The characters make decisions only because the script needs them to, and the emotional payoff feels completely unearned.",
    "The movie seems convinced that louder scenes are automatically more powerful. Instead, the constant shouting and swelling score made the drama feel exhausting.",

    "It is beautifully shot, but almost aggressively boring. Every frame looks composed, yet the story moves with the energy of someone reading a contract.",
    "The film has a decent premise, but it explains the mystery so clumsily that any suspense disappears. Once the big reveal arrives, it feels less shocking than mildly inconvenient.",
    "Some of the supporting actors bring charm, but they are not enough to save a story this unfocused. The film keeps introducing ideas and then abandoning them.",
    "The pacing is a real problem here. Scenes either end too soon to matter or drag on long after the point has been made.",
    "This feels like a movie made from the outline of a better movie. The structure is there, but the emotional detail is missing.",
    "The humor mostly relies on people being loud, awkward, or cruel. After a while, it stops feeling funny and starts feeling lazy.",
    "The lead performance is committed, but the character is written with so little consistency that it is hard to care. One scene contradicts the next, and the film never earns its dramatic turn.",
    "The atmosphere is strong at first, but the film wastes it on a story that becomes increasingly ridiculous. I was hoping for tension and got confusion instead.",
    "The ending tries to reframe everything that came before it, but it only exposes how weak the setup was. A twist cannot fix characters we never understood.",
    "I appreciated the ambition, but the execution is clumsy. It wants to be emotional, stylish, and clever all at once, and ends up being none of those things consistently.",

    "The movie has a strange habit of cutting away just when a scene might become interesting. It feels afraid of its own best moments.",
    "The acting is uneven, with one or two strong performances surrounded by people who seem unsure what tone they are supposed to be playing. The result is distracting rather than immersive.",
    "This is the kind of film where characters explain their feelings in great detail while never actually behaving like real people. It becomes very hard to take seriously.",
    "There is a good short film hidden somewhere inside this overlong feature. Unfortunately, the extra hour adds repetition rather than depth.",
    "The direction feels strangely lifeless. Even scenes that should be tense are staged in such a flat way that they barely create a pulse.",
    "I was surprised by how little momentum the story has. It starts, wanders, pauses, wanders again, and then suddenly ends.",
    "The soundtrack works overtime trying to tell the audience what to feel. The problem is that the script never gives those feelings enough support.",
    "A few clever lines cannot save a film this mechanical. The plot is too neat, the conflicts too convenient, and the characters too forgettable.",
    "The movie looks polished, but it has no real personality. It feels assembled from familiar pieces rather than directed with a clear point of view.",
    "I kept waiting for the emotional connection to click, but it never happened. The film is not awful, just disappointingly cold.",

    "The central relationship is supposed to carry the movie, but there is very little chemistry between the leads. Without that, the whole thing feels strangely hollow.",
    "This film confuses misery with complexity. The characters suffer constantly, but the writing does not make their suffering meaningful.",
    "The plot structure is messy in a way that does not feel intentional. Flashbacks appear at random, side characters vanish, and important choices happen off-screen.",
    "The film has one genuinely moving scene, and then spends the rest of its runtime trying to recreate that feeling without success. It becomes repetitive very quickly.",
    "I could forgive the slow pace if the dialogue had more texture. Instead, the conversations are bland and overly explanatory.",
    "It is hard to stay engaged when the film keeps treating obvious information like a revelation. The suspense never builds because the audience is always two steps ahead.",
    "The visuals are moody, but the story is empty. It is all fog, shadows, and meaningful stares with very little substance underneath.",
    "The film wants to be quirky, but most of the quirks feel calculated. It is charming for about ten minutes, then increasingly irritating.",
    "The lead actor gives a sincere performance, but sincerity alone cannot rescue a weak screenplay. The character arc feels rushed and unconvincing.",
    "The movie is not incompetent, just painfully forgettable. Nothing about it is bad enough to be interesting or good enough to be memorable.",

    "The third act completely undermines the careful setup of the first two. Characters suddenly become foolish, secrets appear from nowhere, and the emotional logic disappears.",
    "I admire films that take their time, but this one simply stalls. The long pauses do not deepen the drama; they just make the thin plot more obvious.",
    "The jokes feel dated and the dramatic scenes feel borrowed from better movies. It never finds its own voice.",
    "The editing makes the story feel choppy and disconnected. Just when a scene starts to settle, the film jumps away without any rhythm.",
    "There is a lot of acting here, but not much truth. Every emotional moment is pushed so hard that it becomes difficult to believe any of it.",
    "The movie sets up an interesting moral dilemma and then solves it in the easiest, least interesting way possible. That was genuinely disappointing.",
    "The film is full of characters making speeches about change, courage, and regret. Somehow none of them actually change in a believable way.",
    "I found the direction oddly anonymous. There are no surprising choices, no memorable images, and no sense of urgency.",
    "The premise could have supported a sharp, tense story, but the script keeps softening every conflict. It becomes bland when it should be gripping.",
    "The film spends so much time building mystery that it forgets to build characters. By the end, I knew the secret but did not care.",

    "The pacing is uneven to the point of distraction. Some scenes rush through important moments, while others linger on details that add nothing.",
    "This movie has the confidence of a masterpiece and the emotional depth of a brochure. It is very pleased with itself, which only makes the flaws harder to ignore.",
    "The supporting cast is more interesting than the lead, which becomes a serious problem. Whenever the story returns to the main character, the energy drops.",
    "The film tries to be dark and mature, but most of its choices feel shallow. Grim lighting and quiet dialogue are not the same thing as complexity.",
    "I wanted more from the ending than a tidy explanation and a sentimental speech. It resolves the plot but not the emotional questions the movie raised.",
    "The story is predictable from the first twenty minutes. That would be fine if the characters were engaging, but they are mostly stereotypes with sad faces.",
    "This felt longer than it actually was, which is never a good sign. The film has too many scenes that repeat information we already understand.",
    "The cinematography is competent, but the action is poorly staged and hard to follow. The camera moves constantly, yet nothing feels exciting.",
    "The film leans heavily on nostalgia, but it does not add much of its own. It feels like a reminder of better movies rather than a strong movie itself.",
    "There are moments of real promise, but they are buried under clumsy exposition. The script never trusts the audience to understand anything on its own.",

    "The emotional climax should have been devastating, but it landed with almost no impact. The film had not done the work to make that moment matter.",
    "The movie seems unsure whether it wants to be realistic or theatrical. That uncertainty makes the performances feel inconsistent and the tone awkward.",
    "I was frustrated by how passive the main character is. Things happen around them, people explain things to them, and then the movie expects growth.",
    "The production design is impressive, but the story inside it feels thin. It is like walking through a beautiful set where nobody has anything interesting to say.",
    "The film makes a lot of noise about big themes, but its actual observations are obvious. It tells us society is complicated and then stops there.",
    "Some scenes are unintentionally funny, which would be more enjoyable if the movie were not clearly asking to be taken seriously. The melodrama becomes too much.",
    "The romantic subplot is especially weak. It feels inserted because the film needed one, not because the characters naturally moved toward each other.",
    "The story takes itself very seriously, but the dialogue often sounds unnatural. People do not talk like this, even in dramatic circumstances.",
    "The movie begins with tension, then spends the next hour slowly draining it away. By the finale, there is almost nothing left to feel.",
    "I respect the craft, but I did not enjoy the film. It is careful, controlled, and emotionally distant in a way that kept me at arm’s length.",

    "The main twist is easy to guess, and the film does not offer enough atmosphere or character work to compensate. It ends up feeling like a puzzle with half the pieces missing.",
    "The director seems more interested in creating pretty images than telling a coherent story. The result is attractive but frustratingly empty.",
    "The film has no sense of escalation. The stakes are supposedly rising, but every scene feels like it has the same emotional temperature.",
    "I found the protagonist more irritating than complex. The film mistakes selfishness for depth and expects sympathy it has not earned.",
    "The comedy is too broad, the drama too thin, and the transitions between them too rough. It feels like two different films fighting for space.",
    "The script keeps introducing side plots that go nowhere. By the end, it feels less like a layered story and more like clutter.",
    "This movie has a polished surface, but very little heart. I admired a few shots and forgot most of the rest almost immediately.",
    "The final act is rushed in a way that makes the previous slow build feel pointless. After waiting so long, the payoff is strangely careless.",
    "The actors try to sell the emotion, but the dialogue keeps getting in their way. Too many lines sound written rather than felt.",
    "It is not the worst film I have seen, but it is one of the more disappointing ones. The ingredients are there, yet the final result is bland.",

    "The film’s atmosphere is heavy, but not in a compelling way. It feels weighed down by its own seriousness.",
    "I kept hoping the movie would surprise me, but it followed the safest path every time. Even the conflicts resolve exactly as expected.",
    "The pacing turns what should be a tense story into a slow crawl. There is a difference between suspense and simply withholding information.",
    "The character motivations are so vague that major decisions feel random. It is difficult to care about consequences when the choices do not make sense.",
    "The movie has a few thoughtful ideas, but it never shapes them into a satisfying drama. It feels more like a draft than a finished film.",
    "The visual style is consistent, but the emotional tone is not. I never felt sure whether the film wanted me to laugh, worry, or cry.",
    "A stronger script might have made this work. As it stands, the film is mostly a slow march toward a conclusion that feels obvious and underwhelming.",
    "The performances are competent, yet no one seems truly alive in the story. Everyone appears to be waiting for the plot to move them along.",
    "The movie tries to end on a poetic note, but it feels more vague than meaningful. Ambiguity only works when there is something solid underneath it.",
    "I left feeling that the film had wasted a good idea. There are flashes of intelligence, but the overall experience is dull and emotionally thin.",

    "The opening promises something sharp and unsettling, but the rest of the movie settles for routine drama. It is disappointing because the potential is obvious.",
    "The script relies too much on coincidence. After the third convenient interruption or discovery, I stopped believing in the world of the film.",
    "The movie is so focused on being tasteful that it forgets to be interesting. Everything is restrained, including my reaction.",
    "The central conflict is introduced clearly enough, but then the film keeps circling it instead of developing it. The result is repetitive and oddly static.",
    "The emotional scenes are shot beautifully, but they feel hollow because the characters are underwritten. A close-up cannot create depth by itself.",
    "The film has a nice sense of place, but the story never rises above familiar beats. It feels like a travel postcard with a weak plot attached.",
    "The ending is meant to feel bittersweet, but it mostly feels incomplete. Too many relationships and ideas are left hanging without purpose.",
    "I did not connect with the film at all. It is polished enough to avoid being terrible, but too cautious to become memorable.",
    "The movie is full of dramatic pauses that seem designed to make ordinary lines sound important. Most of the time, they just make the scenes drag.",
    "There is a decent performance buried in here, but the film around it is slow, predictable, and emotionally unconvincing.",

    "This is a quietly moving film that earns its emotion without begging for it. The performances feel lived-in, and the restrained direction gives every small gesture room to matter.",
    "I was surprised by how much the film stayed with me afterward. It has a gentle pace, but the story keeps building in subtle, meaningful ways.",
    "The acting is excellent across the board, especially in the quieter scenes where so much is communicated without dialogue. It is a patient, thoughtful film with real emotional weight.",
    "The movie balances humor and sadness beautifully. It never feels manipulative, and the characters are flawed in ways that make them more human.",
    "The cinematography is gorgeous without becoming distracting. Every shot seems to support the mood of the story rather than simply showing off.",
    "This film understands pacing in a way many dramas do not. It takes its time, but each scene reveals something new about the characters.",
    "I loved how natural the dialogue felt. People interrupt each other, hesitate, say the wrong thing, and somehow that makes the emotional moments hit harder.",
    "The story is simple on the surface, but it has a quiet depth that sneaks up on you. By the end, I felt genuinely attached to these characters.",
    "The direction is confident and sensitive. It knows when to hold a shot, when to cut away, and when to let silence do the work.",
    "This is not a flashy film, but it is deeply satisfying. The performances, writing, and atmosphere all come together with impressive control.",

    "The film has a wonderful sense of rhythm. Even the slower scenes feel purposeful, and the emotional payoff feels completely earned.",
    "I went in expecting something ordinary and ended up completely absorbed. The characters are written with warmth, complexity, and just enough mystery.",
    "The lead performance is remarkable. It is not loud or showy, but every expression feels precise and emotionally honest.",
    "This movie has the rare ability to be both entertaining and genuinely thoughtful. It respects the audience without becoming cold or overly intellectual.",
    "The script is sharp, but what impressed me most was its restraint. It trusts small moments, and that makes the bigger ones feel much stronger.",
    "The film looks beautiful, but its real strength is the emotional clarity of the storytelling. Nothing feels wasted or accidental.",
    "There is a tenderness to this movie that I found very affecting. It allows its characters to be messy without judging them too harshly.",
    "The pacing is deliberate, yet never dull. Each scene seems to deepen the mood or complicate the relationships in a meaningful way.",
    "I appreciated how the film avoided easy answers. It presents difficult choices with empathy and lets the audience sit with them.",
    "The ending is quiet but powerful. It does not over-explain, and that makes the final image linger even longer.",

    "This is a beautifully acted film with a strong emotional pulse. The story unfolds naturally, and the direction gives the actors space to breathe.",
    "The film has a lived-in atmosphere that makes its world feel real. Even minor characters seem to have lives beyond the frame.",
    "I found the movie unexpectedly funny, but never in a way that undercuts the drama. The humor comes from character, not cheap jokes.",
    "The screenplay is elegant and carefully structured. What seems casual at first slowly reveals itself as emotionally precise.",
    "The cinematography gives the story a rich visual texture. Light, space, and color are used to reflect the characters’ inner lives.",
    "This is the kind of film that rewards attention. Small details in the first half gain new meaning by the end.",
    "The performances are subtle but deeply convincing. No one appears to be acting for a big moment; they simply inhabit the story.",
    "I admired how the film handled conflict without turning everyone into villains. It feels mature, compassionate, and emotionally believable.",
    "The movie is absorbing from the first scene. It has tension, humor, and a strong sense of character without ever feeling forced.",
    "The direction is understated in the best way. It guides the audience gently instead of pushing us toward obvious reactions.",

    "I loved the emotional honesty of this film. It captures regret and hope in a way that feels specific rather than sentimental.",
    "The plot is not especially complicated, but the execution is excellent. The film knows exactly what it is and tells its story with confidence.",
    "There is real chemistry between the leads, and it carries the quieter stretches beautifully. Their relationship feels awkward, warm, and believable.",
    "The film builds suspense through character rather than gimmicks. By the final act, I cared about the outcome because I cared about the people involved.",
    "This movie has a graceful sense of balance. It is sad without being punishing and hopeful without becoming naïve.",
    "The editing is smooth and intelligent. Scenes flow into each other in a way that makes the story feel organic.",
    "I was impressed by how much emotion the film finds in ordinary situations. It makes everyday choices feel meaningful without exaggerating them.",
    "The supporting cast is excellent, giving texture to a story that could have felt small. Everyone adds something distinct.",
    "The dialogue has a natural rhythm that makes the characters feel real. Even simple conversations carry tension and personality.",
    "This is a warm, observant, and surprisingly moving film. It has a quiet confidence that I found very appealing.",

    "The film takes a familiar setup and makes it feel fresh through strong character work. It is less interested in surprises than in emotional truth.",
    "I loved the way the camera lingers on faces just long enough to catch uncertainty. The visual storytelling is delicate but effective.",
    "The movie is funny in a dry, understated way, and that makes the dramatic moments land even harder. It feels honest about how people cope.",
    "The final act is beautifully handled. It brings the story together without feeling too neat or artificial.",
    "The lead actor gives a performance full of small, intelligent choices. You can feel the character thinking before speaking.",
    "This film has atmosphere to spare, but it never lets style overwhelm substance. The mood serves the story perfectly.",
    "I found the pacing immersive rather than slow. The movie lets us settle into the world, and that patience pays off.",
    "The story has real emotional maturity. It understands that people can hurt each other without being cruel, and love each other without knowing how to show it.",
    "The direction is clean and focused, with no unnecessary flourishes. That simplicity gives the film surprising strength.",
    "It is rare to see a film this gentle still feel so engaging. The stakes are personal, but they feel enormous because the characters matter.",

    "The movie creates tension with remarkable restraint. It never shouts, yet I felt completely pulled into the uncertainty of the story.",
    "The cinematography is not just pretty; it tells us how isolated and vulnerable the characters feel. That visual intelligence really elevates the film.",
    "I appreciated the complexity of the main character. The film allows them to be sympathetic, frustrating, and deeply human all at once.",
    "The score is used sparingly, which makes it more effective when it appears. It supports the emotion instead of smothering it.",
    "This is a thoughtful and well-crafted film that grows stronger as it goes. The ending feels quiet, but it lands with real force.",
    "The film has a lovely sense of patience. It does not rush the relationships, so the emotional shifts feel convincing.",
    "I was drawn in by the performances almost immediately. There is a relaxed realism to the acting that makes the story feel intimate.",
    "The screenplay avoids melodrama even when the subject matter gets heavy. That restraint makes the film feel more painful and more truthful.",
    "The movie is beautifully structured, with each scene adding pressure or revealing character. Nothing feels like filler.",
    "This film made me laugh more than I expected and moved me more than I was ready for. That combination is hard to pull off well.",

    "The director has a strong eye for detail. Small objects, rooms, and pauses all seem to carry emotional meaning without being overplayed.",
    "The story unfolds with quiet confidence. It trusts the audience to notice what is changing beneath the surface.",
    "I loved how grounded the performances were. Even the most dramatic moments feel rooted in recognizable human behavior.",
    "The film is visually restrained but emotionally rich. It proves that a modest story can feel powerful when told with care.",
    "The pacing is excellent, especially in the second half where every scene seems to tighten the emotional knot. I was completely invested.",
    "This movie has a generous view of its characters. It notices their mistakes but also their loneliness, fear, and longing.",
    "The film’s humor is subtle and character-based, which makes it feel natural. It never pauses for jokes, yet it is often very funny.",
    "The ending does not provide easy comfort, but it feels honest. I respected that choice and found it surprisingly moving.",
    "The atmosphere is beautifully controlled from beginning to end. The film knows exactly when to be quiet and when to let emotion break through.",
    "This is a smart, intimate, and emotionally satisfying film. It stayed with me because it feels true in small, specific ways.",

    "The film is carried by a wonderful central performance. The actor brings humor, sadness, and quiet dignity to a role that could have been simple.",
    "I liked how the plot reveals itself gradually without relying on cheap twists. The mystery is emotional as much as narrative.",
    "The visuals have a soft, melancholy quality that suits the story perfectly. It is beautiful without feeling artificial.",
    "The film finds drama in hesitation and silence. That may not sound exciting, but it makes the emotional moments feel incredibly real.",
    "The writing is sharp and compassionate. It gives each character a point of view, even when they are making terrible choices.",
    "This movie reminded me how satisfying a well-told simple story can be. Strong acting and careful pacing make all the difference.",
    "The direction is subtle, but the emotional impact is strong. It never overstates its themes, which makes them feel more sincere.",
    "I was impressed by how naturally the film shifts between tension and tenderness. Those tonal changes feel earned rather than abrupt.",
    "The film has a very human understanding of memory and regret. It is sad, but not hopeless.",
    "Every performance feels specific and carefully observed. Even brief scenes have a sense of emotional history behind them.",

    "The movie builds its world through behavior rather than exposition. That makes it feel immersive and refreshingly intelligent.",
    "I found the final scenes deeply affecting. They are quiet, almost modest, but they bring the character arc to a beautiful close.",
    "The film has a strong sense of place, and that setting becomes part of the emotional experience. It feels lived in, not decorative.",
    "The screenplay is honest about how complicated people can be. Nobody is reduced to a lesson or a symbol.",
    "This is a film with patience, warmth, and real craft. It may be understated, but it is never empty.",
    "The camera work is elegant and purposeful. It draws attention to emotion without making the film feel overly designed.",
    "The story is bittersweet in a way that feels earned. I cared about the characters enough to feel the weight of their choices.",
    "The movie is surprisingly gripping for something so quiet. The tension comes from what people cannot say to each other.",
    "I admired the film’s refusal to rush toward easy closure. It gives the audience space to feel the uncertainty.",
    "The performances have a lovely naturalness to them. They make the film feel less like a script and more like captured life.",

    "The film takes its time, but I never felt bored. The emotional details are rich enough to keep every scene alive.",
    "The direction is graceful and confident. It knows that sometimes the most powerful moment is the one held back.",
    "I enjoyed how layered the story became without ever feeling confusing. The structure is careful, but it still feels natural.",
    "The movie has genuine heart, but it avoids becoming overly sweet. That balance makes it feel mature and emotionally satisfying.",
    "The actors bring real warmth to the material. Their chemistry gives the film an intimacy that is hard to fake.",
    "The film is thoughtful, beautifully acted, and quietly funny. It has the kind of emotional detail that makes ordinary scenes memorable.",
    "I loved the way the story slowly shifts our understanding of the characters. By the end, even small earlier moments feel important.",
    "The pacing is calm but precise. It never feels like the film is wandering, only that it is carefully observing.",
    "This is a deeply humane movie. It sees weakness and disappointment clearly, but still leaves room for kindness.",
    "The final image stayed in my mind long after the film ended. It is simple, but it carries the whole emotional journey.",

    "The movie succeeds because it trusts its characters. Instead of forcing drama, it lets conflict grow naturally from who they are.",
    "The cinematography gives the film an intimate, almost tactile quality. You can feel the spaces the characters move through.",
    "The script has a quiet intelligence to it. It says a lot through implication, which makes the viewing experience more rewarding.",
    "I found this film emotionally generous without being sentimental. It understands pain, but it also understands humor and resilience.",
    "The performances are excellent, especially in scenes where nothing dramatic appears to happen. Those are often the moments that reveal the most.",
    "The story is carefully paced and beautifully observed. It builds to a conclusion that feels inevitable but not predictable.",
    "This is a subtle film, but not a slight one. Its emotional force accumulates slowly and then hits harder than expected.",
    "The directing is calm and assured, giving the film a strong sense of emotional control. It never feels rushed or overworked.",
    "I appreciated how the film handled its themes with nuance. It raises difficult questions without turning them into speeches.",
    "The movie left me with a quiet sense of satisfaction. It is well acted, beautifully paced, and emotionally sincere."
   ]

   labels = [
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]

   return texts, labels


def augment_newsgroups_data(target_names):


   texts = [
    "Has anyone here read a good critique of the cosmological argument that does not immediately turn into word games? I am trying to separate the philosophy from the preaching.",
    "I do not mind religious people personally, but I get irritated when public policy is justified with scripture instead of arguments everyone can examine.",
    "The usual claim that morality collapses without belief in God still seems weak to me. People cooperate, care, and reason about harm without needing supernatural enforcement.",
    "I used to accept Pascal's wager when I was younger, but the more I think about it, the more it assumes the exact conclusion it wants.",
    "Is there a FAQ for common atheist responses to arguments from design? I keep seeing the same examples recycled in debates.",

    "I am looking for a simple tool to convert a batch of TIFF files into compressed formats without destroying the color depth. Any recommendations for Unix or DOS?",
    "The renderer works fine on simple objects, but curved surfaces show strange banding after export. Could this be a normals issue or just a bad palette conversion?",
    "For anyone doing texture mapping on older hardware, reducing the image size first seems to help more than changing the dithering method.",
    "Does anyone know a good introductory reference for ray tracing algorithms? I understand the concept, but the acceleration structures still confuse me.",
    "I finally got my viewer to display 24-bit images correctly, but now grayscale files look washed out. I suspect my gamma handling is wrong.",

    "After installing the latest Windows drivers, my system now freezes when closing Program Manager. Has anyone seen this with a Sound Blaster card installed?",
    "Is there a clean way to change default file associations without editing WIN.INI by hand? I keep breaking something whenever I try.",
    "My mouse cursor disappears whenever I switch from a DOS box back to Windows. The machine is usable, but it is getting annoying.",
    "If your Windows setup crashes during printer installation, try removing the old driver files first. That fixed a similar problem on my machine.",
    "Does anyone still have a reliable list of which video drivers work best with Windows 3.1? The one I found seems badly outdated.",

    "I am upgrading an old 486 and cannot decide whether more RAM or a faster IDE drive will make the bigger difference. The machine is mostly used for compiling and games.",
    "My BIOS detects the hard drive only every other boot. Could this be a controller problem, or should I suspect the drive is dying?",
    "For people buying cheap cache chips, check the markings carefully. I have seen boards behave worse with fake or mismatched cache installed.",
    "Does anyone know whether this motherboard supports write-back cache, or is the jumper table in the manual misleading?",
    "The new VGA card works, but only in 8-bit slots, not 16-bit mode. I am starting to think the board has timing issues.",

    "My Quadra occasionally refuses to recognize the external SCSI drive until I power-cycle everything. Is this usually termination, cable length, or the drive itself?",
    "I am thinking of adding more memory to an older LC. Are there specific SIMMs to avoid, or is the Mac fairly forgiving?",
    "The monitor image has a faint shimmer on the left edge. Moving the speakers helped a little, but not completely.",
    "For anyone replacing a PRAM battery, do it before blaming every weird startup problem on the logic board. I learned that the expensive way.",
    "Can a PowerBook battery be rebuilt safely, or is it better to look for a replacement pack? Mine holds a charge for about six minutes.",

    "I am trying to get X running on a new video card, but the server exits with a mode timing error. Does anyone have a working config example?",
    "The application starts correctly over the network, but fonts are huge and ugly on the remote display. I assume this is a font path issue.",
    "Is there a lightweight window manager that does not eat memory on a small workstation? Motif looks nice but feels heavy here.",
    "For anyone fighting mouse problems in X, check the protocol setting before rebuilding the kernel. I wasted an entire evening on that.",
    "How do I make xterm use a different default geometry at startup? The manual page explains everything except the one thing I need.",

    "For sale: 14.4 modem, external, includes power supply and cable. Works fine, only selling because I upgraded.",
    "Does anyone in the Boston area want a used VGA monitor? It has a small scratch on the case, but the picture is still good.",
    "I have a stack of old programming books available for cheap. Prefer local pickup, since shipping will probably cost more than the books.",
    "Looking to buy a used SCSI hard drive around 500 MB. Please email condition, brand, and asking price.",
    "Selling a pair of speakers and an ISA sound card as a bundle. Not fancy, but good enough for a spare PC.",

    "My car has started pulling slightly to the right under braking. Tires look fine, so I am wondering if this is a caliper or alignment issue.",
    "I test drove a small sedan yesterday and was surprised by how vague the steering felt. The engine was decent, but the suspension did not inspire confidence.",
    "For winter driving, good tires matter more than most of the gadgets people argue about. I learned that after one very slow slide into a curb.",
    "Does anyone have experience with high-mileage synthetic oil in older engines? I am not expecting miracles, just fewer cold-start noises.",
    "The dealer says the timing belt can wait another year, but the manual says otherwise. I think I would rather spend the money than risk the engine.",

    "My bike hesitates when I roll on the throttle from low RPM. Could dirty carbs cause that, or should I look at ignition first?",
    "I finally bought proper riding boots, and now I wonder why I waited so long. They are less comfortable off the bike, but much better while riding.",
    "Anyone have advice for a first long highway trip on a smaller motorcycle? I am mostly worried about fatigue and crosswinds.",
    "The chain adjustment in the manual seems too tight compared with what experienced riders recommend. Which one should I trust?",
    "Lane position makes a bigger difference than new riders think. Being visible is often more important than being technically correct.",

    "The bullpen lost that game more than the starter did. You cannot walk the leadoff man in the eighth and expect good things.",
    "I still think on-base percentage is underrated by casual fans. A flashy batting average does not help much if the guy never walks.",
    "Does anyone have minor league stats for that rookie call-up? The broadcast made him sound like a future star, but I want to see the numbers.",
    "That was one of the strangest managerial decisions I have seen this season. Pulling a pitcher after two soft singles felt like panic.",
    "The division race is tighter than people expected. A few injuries and suddenly the whole schedule looks different.",

    "The goalie kept them alive for two periods, but the defense completely collapsed in the third. You cannot give up that many odd-man rushes.",
    "I love a physical game, but some of those hits were just reckless. The league needs to be more consistent with suspensions.",
    "Does anyone know if the rookie center is staying on the top line? His passing has been better than his point totals suggest.",
    "Power play entries are killing this team. They spend half the advantage dumping the puck in and losing the race.",
    "That overtime goal was a beauty. Quick release, perfect screen, and the goalie never saw it until it was behind him.",

    "Can someone explain whether key escrow actually improves security, or does it just create another target for attackers?",
    "I am trying to understand the practical difference between authentication and encryption in a mail system. Most articles blur the two together.",
    "The proposed restrictions on strong encryption seem shortsighted. Weakening civilian systems will not magically stop criminals from using private tools.",
    "For casual users, public key cryptography still feels too hard to manage. The math is elegant, but the key handling is where people fail.",
    "Has anyone tested the performance of RSA signatures on older workstations? I am curious how practical it is for frequent message signing.",

    "I need help identifying a resistor that has a burned color band. The circuit is a small power supply, and the schematic is missing.",
    "My homemade amplifier works, but there is a low hum even with no input connected. Is grounding the most likely culprit?",
    "For beginners building power supplies, please remember that large capacitors can stay charged after unplugging. A cheap meter is not optional.",
    "Does anyone have a simple explanation of why my op-amp saturates when the input signal looks perfectly reasonable?",
    "I replaced the voltage regulator and the board now runs cooler, but the output still drifts under load. Maybe the transformer is undersized.",

    "Has anyone had lingering fatigue after a viral infection even after normal blood tests? I am not asking for diagnosis, just experiences.",
    "The article makes the treatment sound miraculous, but the sample size is tiny. I wish medical reporting included more context about study design.",
    "My doctor suggested keeping a symptom diary before changing medication. It sounds boring, but it may help separate patterns from random bad days.",
    "Can someone explain the difference between sensitivity and specificity in plain language? I keep mixing them up when reading test results.",
    "Please be careful with herbal remedies if you are already taking prescriptions. Natural does not automatically mean harmless.",

    "The latest launch delay is frustrating, but I would rather see a scrub than a failure. Rockets are not exactly forgiving machines.",
    "Does anyone know how much payload capacity is lost when a mission requires a high-inclination orbit? I am trying to understand the tradeoff.",
    "The images from the probe are incredible, but the engineering behind getting them back to Earth is just as impressive.",
    "I still think a permanent lunar base will require boring infrastructure before anything glamorous happens. Power, dust control, and logistics come first.",
    "Can someone recommend a good beginner book on orbital mechanics? I can follow the basic idea, but transfers still make my head hurt.",

    "I have been thinking about forgiveness lately, especially when the other person never admits wrongdoing. How do Christians here understand that tension?",
    "Our church is discussing how to welcome newcomers without overwhelming them. Small groups seem helpful, but they can also feel closed from the outside.",
    "Prayer has not always changed my circumstances, but it has often changed how I carry them. I wonder if that is part of what people mean by peace.",
    "Does anyone know a good study guide for the Gospel of Luke? I want something thoughtful but not overly academic.",
    "The sermon this week focused on humility, and it struck me how easily religious language can become pride in disguise.",

    "The debate always seems to ignore responsible owners who already follow the law. Punishing them for criminal behavior by others makes no sense to me.",
    "Background checks are worth discussing, but vague bans based on appearance are bad policy. Define the mechanical features or do not write the law.",
    "Has anyone seen reliable numbers on defensive gun use that are not from advocacy groups? Every source I find seems to have an agenda.",
    "Storage laws may be one area where both sides could actually talk. Keeping firearms away from children should not be controversial.",
    "The media coverage after every shooting becomes predictable within minutes. Lots of emotion, very little technical accuracy.",

    "The discussion of the peace process keeps ignoring what ordinary people on both sides are living through. Maps and slogans do not capture that reality.",
    "I wish posters would distinguish between criticizing a government and attacking an entire people. The two get mixed together constantly here.",
    "Does anyone have a good historical timeline of the border proposals? I am tired of arguments that start in the middle and pretend nothing came before.",
    "The latest speech sounded tough, but I do not see how it changes conditions on the ground. Security and dignity both have to be addressed.",
    "Every thread on this topic turns into competing tragedies. A serious discussion should be able to recognize more than one kind of suffering.",

    "The tax proposal sounds simple in the press release, but the details shift costs in ways most voters will not notice until later.",
    "I am tired of politicians calling every compromise a betrayal. Governing is supposed to involve tradeoffs, not permanent campaign slogans.",
    "Does anyone know where to find the full text of the bill being debated? News summaries are leaving out the parts I actually care about.",
    "The scandal is ugly, but I would rather see evidence than another week of speculation. Outrage is not a substitute for facts.",
    "Term limits sound attractive, but they may just give lobbyists more power over inexperienced legislators. I am not sure the cure is better than the disease.",

    "I have a question about how different traditions interpret prophecy. It seems like everyone claims the plain reading until the plain reading becomes inconvenient.",
    "Religious debates here would be more useful if people defined what they mean by faith before attacking it. The word carries too many assumptions.",
    "I am not convinced that mystical experiences prove doctrine, but I also do not think they can be dismissed as simple nonsense.",
    "Does anyone have recommendations for comparative religion books that are respectful but not devotional? I want analysis, not conversion material.",
    "The argument over miracles always seems to become an argument over what counts as evidence. That may be the real disagreement underneath everything."
   ]

   labels = [
    target_names.index("alt.atheism"),
    target_names.index("alt.atheism"),
    target_names.index("alt.atheism"),
    target_names.index("alt.atheism"),
    target_names.index("alt.atheism"),

    target_names.index("comp.graphics"),
    target_names.index("comp.graphics"),
    target_names.index("comp.graphics"),
    target_names.index("comp.graphics"),
    target_names.index("comp.graphics"),

    target_names.index("comp.os.ms-windows.misc"),
    target_names.index("comp.os.ms-windows.misc"),
    target_names.index("comp.os.ms-windows.misc"),
    target_names.index("comp.os.ms-windows.misc"),
    target_names.index("comp.os.ms-windows.misc"),

    target_names.index("comp.sys.ibm.pc.hardware"),
    target_names.index("comp.sys.ibm.pc.hardware"),
    target_names.index("comp.sys.ibm.pc.hardware"),
    target_names.index("comp.sys.ibm.pc.hardware"),
    target_names.index("comp.sys.ibm.pc.hardware"),

    target_names.index("comp.sys.mac.hardware"),
    target_names.index("comp.sys.mac.hardware"),
    target_names.index("comp.sys.mac.hardware"),
    target_names.index("comp.sys.mac.hardware"),
    target_names.index("comp.sys.mac.hardware"),

    target_names.index("comp.windows.x"),
    target_names.index("comp.windows.x"),
    target_names.index("comp.windows.x"),
    target_names.index("comp.windows.x"),
    target_names.index("comp.windows.x"),

    target_names.index("misc.forsale"),
    target_names.index("misc.forsale"),
    target_names.index("misc.forsale"),
    target_names.index("misc.forsale"),
    target_names.index("misc.forsale"),

    target_names.index("rec.autos"),
    target_names.index("rec.autos"),
    target_names.index("rec.autos"),
    target_names.index("rec.autos"),
    target_names.index("rec.autos"),

    target_names.index("rec.motorcycles"),
    target_names.index("rec.motorcycles"),
    target_names.index("rec.motorcycles"),
    target_names.index("rec.motorcycles"),
    target_names.index("rec.motorcycles"),

    target_names.index("rec.sport.baseball"),
    target_names.index("rec.sport.baseball"),
    target_names.index("rec.sport.baseball"),
    target_names.index("rec.sport.baseball"),
    target_names.index("rec.sport.baseball"),

    target_names.index("rec.sport.hockey"),
    target_names.index("rec.sport.hockey"),
    target_names.index("rec.sport.hockey"),
    target_names.index("rec.sport.hockey"),
    target_names.index("rec.sport.hockey"),

    target_names.index("sci.crypt"),
    target_names.index("sci.crypt"),
    target_names.index("sci.crypt"),
    target_names.index("sci.crypt"),
    target_names.index("sci.crypt"),

    target_names.index("sci.electronics"),
    target_names.index("sci.electronics"),
    target_names.index("sci.electronics"),
    target_names.index("sci.electronics"),
    target_names.index("sci.electronics"),

    target_names.index("sci.med"),
    target_names.index("sci.med"),
    target_names.index("sci.med"),
    target_names.index("sci.med"),
    target_names.index("sci.med"),

    target_names.index("sci.space"),
    target_names.index("sci.space"),
    target_names.index("sci.space"),
    target_names.index("sci.space"),
    target_names.index("sci.space"),

    target_names.index("soc.religion.christian"),
    target_names.index("soc.religion.christian"),
    target_names.index("soc.religion.christian"),
    target_names.index("soc.religion.christian"),
    target_names.index("soc.religion.christian"),

    target_names.index("talk.politics.guns"),
    target_names.index("talk.politics.guns"),
    target_names.index("talk.politics.guns"),
    target_names.index("talk.politics.guns"),
    target_names.index("talk.politics.guns"),

    target_names.index("talk.politics.mideast"),
    target_names.index("talk.politics.mideast"),
    target_names.index("talk.politics.mideast"),
    target_names.index("talk.politics.mideast"),
    target_names.index("talk.politics.mideast"),

    target_names.index("talk.politics.misc"),
    target_names.index("talk.politics.misc"),
    target_names.index("talk.politics.misc"),
    target_names.index("talk.politics.misc"),
    target_names.index("talk.politics.misc"),

    target_names.index("talk.religion.misc"),
    target_names.index("talk.religion.misc"),
    target_names.index("talk.religion.misc"),
    target_names.index("talk.religion.misc"),
    target_names.index("talk.religion.misc")
   ]

   return texts, labels



# 4. Model: Embedding + Conv1D + Linear + Linear


class SimpleCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 num_conv_layers=1, activation_fn=nn.ReLU):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        layers = []
        in_channels = embed_dim

        for _ in range(num_conv_layers):
            layers.append(nn.Conv1d(in_channels, 128, kernel_size=3, padding=1))
            layers.append(activation_fn())
            in_channels = 128

        self.conv = nn.Sequential(*layers)

        self.fc1 = nn.Linear(128, 64)
        self.act = activation_fn()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = torch.max(x, dim=2).values
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x



# 5. Accuracy


def compute_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = torch.argmax(model(X), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total



# 6. Training


def train_model(model, train_loader, val_loader=None, epochs=10):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)

        msg = f"Epoch {epoch + 1}, loss={total_loss / len(train_loader.dataset):.4f}"

        if val_loader:
            acc = compute_accuracy(model, val_loader)
            msg += f", val_acc={acc:.4f}"

        print(msg)



# 7. IMDB


def run_imdb(num_conv_layers, activation_fn):
    print("Loading IMDB...")

    imdb = load_dataset("imdb")
    texts = imdb["train"]["text"]
    labels = imdb["train"]["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    aug_texts, aug_labels = augment_imdb_data()
    X_train = list(X_train) + aug_texts
    y_train = list(y_train) + aug_labels

    vocab = build_vocab(X_train)

    train_ds = TextDataset(X_train, y_train, vocab)
    test_ds = TextDataset(X_test, y_test, vocab)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = SimpleCNN(
        vocab_size=len(vocab),
        embed_dim=100,
        num_classes=2,
        num_conv_layers=num_conv_layers,
        activation_fn=activation_fn
    )

    train_model(model, train_loader, val_loader=test_loader, epochs=10)
    acc = compute_accuracy(model, test_loader)

    print(f"IMDB Test Accuracy: {acc:.4f}")
    return acc



# 8. 20 Newsgroups


def run_newsgroups(num_conv_layers, activation_fn):
    print("Loading 20 Newsgroups...")

    data = fetch_20newsgroups(subset="train")
    texts = data.data
    labels = data.target
    target_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    aug_texts, aug_labels = augment_newsgroups_data(target_names)
    X_train = list(X_train) + aug_texts
    y_train = list(y_train) + aug_labels

    vocab = build_vocab(X_train)

    train_ds = TextDataset(X_train, y_train, vocab)
    test_ds = TextDataset(X_test, y_test, vocab)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = SimpleCNN(
        vocab_size=len(vocab),
        embed_dim=100,
        num_classes=20,
        num_conv_layers=num_conv_layers,
        activation_fn=activation_fn
    )

    train_model(model, train_loader, val_loader=test_loader, epochs=10)
    acc = compute_accuracy(model, test_loader)

    print(f"20 Newsgroups Test Accuracy: {acc:.4f}")
    return acc



# 9. Main Experiment


def experiment():
    results = []

    configs = [
        (1, nn.ReLU),
        (2, nn.ReLU),
        (3, nn.ReLU),
        (1, nn.LeakyReLU),
        (1, nn.GELU),
    ]

    for num_layers, act in configs:
        print(f"\nRunning IMDB: {num_layers} conv layers, {act.__name__}")
        acc = run_imdb(num_layers, act)
        results.append({
            "Dataset": "IMDB",
            "Conv Layers": num_layers,
            "Activation": act.__name__,
            "Accuracy": acc
        })

    for num_layers, act in configs:
        print(f"\nRunning 20NG: {num_layers} conv layers, {act.__name__}")
        acc = run_newsgroups(num_layers, act)
        results.append({
            "Dataset": "20NG",
            "Conv Layers": num_layers,
            "Activation": act.__name__,
            "Accuracy": acc
        })

    df = pd.DataFrame(results)
    print("\nFINAL RESULTS:")
    print(df)

# 10. MAIN

if __name__ == "__main__":
    experiment()

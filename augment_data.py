# ==========================================================
# PROMPTY DO GENEROWANIA AUGMENTACJI (200 IMDB + 200 20NG)
# ==========================================================

PROMPT_IMDB_200 = """
Generate 200 additional movie reviews in the style of the IMDB dataset.

REQUIREMENTS:
- Create exactly 100 negative reviews and 100 positive reviews.
- Each review must be 4–12 sentences long.
- Style MUST match real IMDB user reviews:
  - emotional, chaotic, personal, opinionated
  - slang, typos, CAPS LOCK, repetition
  - messy, unpolished, sometimes ranting
- Avoid sounding like a professional critic.
- Avoid generic phrasing.
- Do NOT mention that the text is generated.
- Do NOT include movie titles or copyrighted content.

OUTPUT FORMAT:
Return a Python list of tuples:

[
    ("review text 1", 0),
    ("review text 2", 1),
    ...
]

Where:
0 = negative  
1 = positive

IMPORTANT:
- The output MUST be valid Python syntax.
- The list MUST contain exactly 200 tuples.
- Each tuple MUST contain (string, integer).
"""

PROMPT_20NG_200 = """
Generate additional training data for the 20 Newsgroups dataset.

REQUIREMENTS:
- For EACH of the 20 classes, generate EXACTLY 10 new examples.
- Total: 200 samples.
- Each sample must look like a real Usenet/forum post:
  - informal, short, messy
  - sometimes technical, sometimes opinionated
  - includes questions, complaints, advice, reactions
- Length: 1–6 sentences.
- Style MUST match the original dataset: casual, unpolished, realistic.
- Do NOT include copyrighted content.

CLASSES:
alt.atheism, comp.graphics, comp.os.ms-windows.misc,
comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x,
misc.forsale, rec.autos, rec.motorcycles, rec.sport.baseball,
rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space,
soc.religion.christian, talk.politics.guns, talk.politics.mideast,
talk.politics.misc, talk.religion.misc.

OUTPUT FORMAT:
Return a Python list of tuples:

[
    ("text sample 1", "alt.atheism"),
    ("text sample 2", "comp.graphics"),
    ...
]

IMPORTANT:
- The output MUST be valid Python syntax.
- The list MUST contain exactly 200 tuples.
- Each tuple MUST contain (string, string).
"""

# ==========================================================
# MIEJSCE NA TWOJE DANE AUGMENTACYJNE
# ==========================================================

# Wklej tutaj wynik PROMPT_IMDB_200
imdb_augmented = [
   ("I honestly didn’t expect much, but the movie hooked me right away. The characters feel like real people instead of cardboard stereotypes. The emotional beats land without being forced, and the humor is surprisingly natural. The pacing is slow in a good way, letting scenes breathe. By the end I felt weirdly attached to everyone. It’s not perfect, but it’s warm and sincere in a way that stuck with me.", 1),

   ("This film has such a gentle confidence. It doesn’t rush, doesn’t shout, doesn’t try to impress you with flashy nonsense. Instead it trusts the story and the actors, and it works beautifully. The quiet scenes hit the hardest. The ending left me smiling like an idiot. Honestly one of the nicest surprises I’ve had in a while.", 1),

   ("I loved how grounded everything felt. Even the dramatic moments have this subtle realism that makes them hit harder. The cinematography is simple but effective, and the soundtrack fits perfectly. The chemistry between the leads is fantastic. I didn’t expect to care this much, but here we are.", 1),

   ("The movie builds emotion slowly, and when it finally hits, it hits hard. The performances are honest and vulnerable. The writing avoids clichés and lets the characters make believable mistakes. I found myself thinking about certain scenes long after the credits rolled. A genuinely touching film.", 1),

   ("This was way better than I expected. The directing is subtle but smart, and the actors carry the story with ease. The humor is soft and human, not forced. The emotional payoff in the last act is absolutely worth the wait. A small film with a big heart.", 1),

   ("The film feels incredibly human. The characters are flawed but lovable, and the story treats them with compassion. The pacing is slow but intentional, and the emotional moments feel earned. I didn’t expect to tear up, but here we are. Beautiful little movie.", 1),

   ("The acting is the standout here. Every expression, every pause feels meaningful. The story is simple but told with so much care. The cinematography is understated but gorgeous. I walked away feeling strangely uplifted. Highly recommended.", 1),

   ("This movie surprised me with how emotionally rich it is. The writing is sharp, the characters feel lived-in, and the direction is confident without being flashy. The final scene absolutely wrecked me in the best way. A quiet gem.", 1),

   ("The film balances humor and sadness so well. Nothing feels exaggerated or manipulative. The performances are warm and believable, and the story unfolds naturally. I didn’t expect to enjoy it this much, but it completely won me over.", 1),

   ("A genuinely heartfelt movie. The pacing is slow but never boring, and the emotional beats feel honest. The cast has great chemistry, and the directing is subtle but effective. I walked away feeling lighter. Lovely film.", 1),

   ("This movie has such a comforting vibe. Even when things get heavy, it never feels hopeless. The characters are written with empathy, and the performances are fantastic. The ending is bittersweet in the best way. Really enjoyed this.", 1),

   ("The film is beautifully understated. No melodrama, no cheap twists, just honest storytelling. The acting is phenomenal, especially in the quieter scenes. The emotional payoff is subtle but powerful. A really touching experience.", 1),

   ("I didn’t expect to get emotional, but the movie got me good. The writing is thoughtful, the pacing is patient, and the performances are incredibly grounded. The final act is especially strong. A lovely surprise.", 1),

   ("This is the kind of movie that sneaks up on you. It starts slow, but by the halfway point I was completely invested. The characters feel real, the dialogue is natural, and the emotional moments land perfectly. Highly recommended.", 1),

   ("The directing is so confident without being showy. The actors deliver incredibly natural performances, and the story unfolds with quiet grace. The emotional beats feel earned, not forced. A genuinely beautiful film.", 1),

   ("The film has a soft, melancholic tone that works perfectly. The performances are subtle but powerful, and the writing is thoughtful. The ending is bittersweet and incredibly moving. Loved it.", 1),

   ("This movie is full of small, beautiful moments. The acting is fantastic, the writing is sharp, and the pacing is perfect for the story it wants to tell. I didn’t expect to enjoy it this much, but it really stuck with me.", 1),

   ("The chemistry between the leads is incredible. Every scene feels alive and honest. The directing is subtle but effective, and the emotional payoff is fantastic. A genuinely touching film.", 1),

   ("The film handles emotion with such care. Nothing feels exaggerated or manipulative. The performances are grounded, and the writing is thoughtful. The ending hit me harder than I expected. Beautiful movie.", 1),

   ("This movie is such a pleasant surprise. The pacing is slow but intentional, the acting is fantastic, and the emotional moments feel incredibly real. I walked away feeling genuinely moved.", 1),

   ("The story is simple but told with so much heart. The performances are warm and believable, and the directing is subtle but effective. The final scene is absolutely beautiful. Loved it.", 1),

   ("The film has a quiet confidence that I really appreciated. The acting is phenomenal, the writing is sharp, and the emotional beats feel earned. A genuinely touching experience.", 1),

   ("This movie is full of warmth and humanity. The characters feel real, the dialogue is natural, and the emotional moments land perfectly. I didn’t expect to enjoy it this much.", 1),

   ("The pacing is slow but perfect for the story. The acting is incredibly natural, and the emotional payoff is fantastic. A small but beautiful film.", 1),

   ("A genuinely heartfelt movie with fantastic performances and thoughtful writing. The emotional moments feel honest and earned. Highly recommended.", 1),

    ("I honestly don’t know what the director was trying to do here. The pacing is all over the place, scenes drag forever, and then suddenly everything happens at once. The acting feels stiff, like everyone is reading cue cards. The emotional moments fall completely flat. I kept waiting for something to click, but it never did. Just a frustrating experience.", 0),

("This movie is a total mess. The plot makes no sense, characters disappear for half an hour, and the editing feels like it was done in the back of a moving car. The dramatic scenes are laughably bad. I checked the time at least five times. Nothing lands, nothing works, nothing matters. Painful.", 0),

("I tried so hard to like this, but wow… it’s rough. The dialogue is wooden, the acting is awkward, and the story feels like a first draft someone forgot to rewrite. The emotional beats are so forced it’s almost embarrassing. I wanted to turn it off halfway through.", 0),

("The movie looks cheap, sounds cheap, and honestly feels cheap. The lighting is awful, the camera work is shaky for no reason, and the soundtrack is weirdly loud in random places. The story goes nowhere. I kept waiting for something interesting to happen, but nope.", 0),

("This was exhausting to sit through. The pacing is painfully slow, the characters are boring, and the plot is basically nonexistent. Every scene feels like filler. The ending tries to be deep but just comes off as pretentious. Total waste of time.", 0),

("I don’t know who approved this script, but they should apologize. The dialogue is cringe, the jokes don’t land, and the emotional scenes feel like a parody. The actors look uncomfortable the entire time. I wanted it to end so badly.", 0),

("The movie tries so hard to be meaningful, but it ends up being boring and confusing. The symbolism is so on-the-nose it’s almost funny. The pacing is glacial. I kept zoning out and honestly didn’t miss anything. Just bad.", 0),

("This film is a disaster. The story is incoherent, the characters are unlikable, and the acting is painfully stiff. The cinematography looks like it was shot on a phone. I can’t believe I wasted two hours on this.", 0),

("I swear the movie was written by someone who has never met another human being. Nobody talks like this. Nobody acts like this. The emotional scenes are so forced I actually laughed. Just terrible writing all around.", 0),

("The plot is full of holes, the pacing is awful, and the acting is shockingly bad. The movie keeps pretending something important is happening, but nothing ever does. The ending is especially ridiculous. I regret watching this.", 0),

("This movie is so boring it should be illegal. Every scene drags on forever. The characters have no depth, the story has no direction, and the dialogue has no life. I kept hoping it would get better, but it only got worse.", 0),

("The film tries to be artistic but ends up being pretentious nonsense. Long shots of nothing happening, characters staring into space, and dialogue that sounds like a bad poetry slam. I couldn’t take it seriously.", 0),

("I don’t know what genre this was supposed to be, but it fails at all of them. Not funny, not dramatic, not interesting. The pacing is awful and the acting is even worse. I wanted to turn it off after 20 minutes.", 0),

("The editing is so choppy it gave me a headache. Scenes start and end abruptly, characters teleport around, and the tone shifts randomly. It feels unfinished. Honestly one of the worst edited films I’ve ever seen.", 0),

("This movie is just noise. Loud music, loud acting, loud everything. But none of it means anything. The story is paper-thin and the characters are walking clichés. I was bored and annoyed the entire time.", 0),

("The film thinks it’s deep, but it’s really just confusing and dull. Characters talk in vague metaphors that don’t mean anything. The pacing is painfully slow. I kept waiting for the point, but there wasn’t one.", 0),

("I can’t believe how bad this was. The acting is wooden, the story is predictable, and the emotional scenes feel fake. The whole movie feels like a student project that somehow got released. Just awful.", 0),

("The movie drags on forever with nothing happening. The characters are bland, the dialogue is repetitive, and the plot is basically nonexistent. The ending tries to be profound but just feels silly. Total snoozefest.", 0),

("This film is a perfect example of style over substance. Lots of fancy shots, but no actual story. The characters are shallow, the pacing is terrible, and the acting is mediocre at best. I was bored out of my mind.", 0),

("The writing is unbelievably bad. Characters make the dumbest decisions imaginable just to move the plot forward. The emotional scenes are laughably overacted. I couldn’t take any of it seriously.", 0),

("The movie feels like it was made by someone who hates movies. No tension, no emotion, no direction. Just a bunch of scenes thrown together. I kept hoping it would end. Miserable experience.", 0),

("This was honestly painful to watch. The pacing is awful, the acting is stiff, and the story goes nowhere. The movie keeps pretending something important is happening, but nothing ever does. Just bad.", 0),

("The film is so slow and dull that I almost fell asleep. The characters are boring, the dialogue is flat, and the plot is nonexistent. I kept waiting for something to happen, but it never did.", 0),

("This movie is a complete waste of time. The acting is terrible, the writing is worse, and the pacing is unbearable. I regret watching it. Easily one of the worst films I’ve seen this year.", 0),

("I don’t know how they managed to make something this boring. The story is predictable, the characters are bland, and the emotional moments feel fake. I couldn’t wait for it to be over.", 0),

("I didn’t expect much from this movie, but it completely won me over. The characters feel real, the dialogue is natural, and the emotional moments hit surprisingly hard. The pacing is slow in a comforting way. By the end I felt genuinely attached to everyone. A small but beautiful film.", 1),

("This movie has such a warm, human vibe. The acting is incredibly natural, and the story unfolds with a quiet confidence. Nothing feels forced. The emotional payoff in the last act is subtle but powerful. I walked away smiling.", 1),

("The film surprised me with how honest it felt. The performances are grounded, the writing is thoughtful, and the directing is gentle but effective. The emotional scenes land perfectly without being melodramatic. A lovely experience.", 1),

("I loved how the movie lets its characters breathe. No rushed scenes, no unnecessary drama. Just honest storytelling. The chemistry between the leads is fantastic. The ending is bittersweet in the best way.", 1),

("This film is full of small, beautiful moments. The acting is fantastic, the writing is sharp, and the pacing is perfect for the story. I didn’t expect to enjoy it this much, but it really stuck with me.", 1),

("The movie has such a comforting tone. Even when things get heavy, it never feels hopeless. The characters are written with empathy, and the performances are incredibly natural. A genuinely touching film.", 1),

("The directing is subtle but confident. The actors deliver incredibly grounded performances, and the story unfolds with quiet grace. The emotional beats feel earned, not forced. A beautiful little film.", 1),

("This movie is such a pleasant surprise. The pacing is slow but intentional, the acting is fantastic, and the emotional moments feel incredibly real. I walked away feeling genuinely moved.", 1),

("The film balances humor and sadness so well. Nothing feels exaggerated or manipulative. The performances are warm and believable, and the story unfolds naturally. A genuinely heartfelt movie.", 1),

("I didn’t expect to get emotional, but the movie got me good. The writing is thoughtful, the pacing is patient, and the performances are incredibly grounded. The final act is especially strong. Loved it.", 1),

("The chemistry between the leads is incredible. Every scene feels alive and honest. The directing is subtle but effective, and the emotional payoff is fantastic. A genuinely touching film.", 1),

("The film handles emotion with such care. Nothing feels exaggerated or manipulative. The performances are grounded, and the writing is thoughtful. The ending hit me harder than I expected. Beautiful movie.", 1),

("This movie is full of warmth and humanity. The characters feel real, the dialogue is natural, and the emotional moments land perfectly. I didn’t expect to enjoy it this much.", 1),

("The pacing is slow but perfect for the story. The acting is incredibly natural, and the emotional payoff is fantastic. A small but beautiful film.", 1),

("The film has a quiet confidence that I really appreciated. The acting is phenomenal, the writing is sharp, and the emotional beats feel earned. A genuinely touching experience.", 1),

("This movie is full of soft, emotional moments that feel incredibly real. The performances are subtle but powerful, and the writing is thoughtful. The ending is bittersweet and beautiful.", 1),

("The story is simple but told with so much heart. The performances are warm and believable, and the directing is subtle but effective. The final scene is absolutely beautiful. Loved it.", 1),

("The film has such a gentle, melancholic tone. The acting is fantastic, the writing is thoughtful, and the emotional moments feel incredibly honest. A lovely surprise.", 1),

("I didn’t expect to enjoy this as much as I did. The characters are incredibly well-written, the pacing is patient, and the emotional payoff is fantastic. A genuinely heartfelt movie.", 1),

("The movie feels incredibly human. The characters are flawed but lovable, and the story treats them with compassion. The emotional moments feel earned. A beautiful little film.", 1),

("The acting is the standout here. Every expression, every pause feels meaningful. The story is simple but told with so much care. I walked away feeling strangely uplifted.", 1),

("This film surprised me with how emotionally rich it is. The writing is sharp, the characters feel lived-in, and the direction is confident without being flashy. A quiet gem.", 1),

("The movie builds emotion slowly, and when it finally hits, it hits hard. The performances are honest and vulnerable. The writing avoids clichés and lets the characters make believable mistakes.", 1),

("The film balances humor and sadness beautifully. The characters feel real, the dialogue is natural, and the emotional moments land perfectly. A genuinely touching experience.", 1),

("A genuinely heartfelt movie with fantastic performances and thoughtful writing. The emotional moments feel honest and earned. Highly recommended.", 1),
("I honestly don’t know how a movie can be this boring. Every scene drags on forever, the characters have zero personality, and the plot feels like it was made up on the spot. The emotional moments are so forced it’s almost funny. I kept checking the time, hoping it would end soon.", 0),

("This movie is a complete disaster. The pacing is awful, the acting is stiff, and the writing is unbelievably lazy. Characters make random decisions just to move the plot forward. Nothing feels natural. I regret watching this.", 0),

("The film tries so hard to be deep, but it’s just confusing and dull. Long scenes of people staring into space don’t make a movie meaningful. The dialogue is painfully awkward. I couldn’t wait for it to be over.", 0),

("I don’t know what the director was thinking. The tone is all over the place, the acting is wooden, and the story goes nowhere. The emotional scenes feel fake and the humor is cringe. Just a bad movie.", 0),

("This was honestly painful to sit through. The pacing is glacial, the characters are bland, and the plot is basically nonexistent. The movie keeps pretending something important is happening, but nothing ever does. Total waste of time.", 0),

("The editing is so bad it’s distracting. Scenes cut abruptly, characters teleport around, and the pacing is completely broken. The acting doesn’t help either. Everyone looks like they want to go home. Terrible experience.", 0),

("This movie is a perfect example of wasted potential. The idea could’ve worked, but the execution is awful. The writing is lazy, the acting is flat, and the directing is clueless. I was bored the entire time.", 0),

("I swear the movie was written in one night. The dialogue is unnatural, the characters are inconsistent, and the plot is full of holes. The emotional scenes are laughably bad. I couldn’t take any of it seriously.", 0),

("The film is so slow and dull that I almost fell asleep. The characters are boring, the dialogue is repetitive, and the story goes nowhere. The ending tries to be profound but just feels silly. Miserable watch.", 0),

("This movie is just noise. Loud music, loud acting, loud everything. But none of it means anything. The story is paper-thin and the characters are walking clichés. I was annoyed the entire time.", 0),

("The acting is unbelievably bad. Everyone sounds like they’re reading lines off a cue card. The emotional scenes are so overacted they become unintentionally funny. I couldn’t stop cringing.", 0),

("The movie tries to be artistic but ends up being pretentious nonsense. Long shots of nothing happening, characters talking in vague metaphors, and a plot that goes nowhere. I hated every minute.", 0),

("This film is a mess. The pacing is awful, the writing is lazy, and the acting is stiff. The emotional moments feel fake and the humor doesn’t land. I wanted to turn it off halfway through.", 0),

("I don’t know how they managed to make something this boring. The story is predictable, the characters are bland, and the emotional moments feel forced. I couldn’t wait for it to end.", 0),

("The cinematography looks cheap, the lighting is awful, and the sound mix is inconsistent. The story is confusing and the acting is wooden. Nothing about this movie works. Just bad.", 0),

("This movie is so slow it feels like time stops. The characters are uninteresting, the dialogue is flat, and the plot is nonexistent. The ending tries to be emotional but completely fails.", 0),

("The writing is unbelievably lazy. Characters make dumb decisions just to move the plot forward. The emotional scenes are laughably overacted. I couldn’t take any of it seriously.", 0),

("The film feels unfinished. Scenes drag on forever, the pacing is awful, and the acting is stiff. The story goes nowhere and the ending is ridiculous. I regret watching this.", 0),

("This movie is a complete waste of time. The acting is terrible, the writing is worse, and the pacing is unbearable. I kept hoping it would get better, but it only got worse.", 0),

("The plot is full of holes, the pacing is awful, and the acting is shockingly bad. The movie keeps pretending something important is happening, but nothing ever does. Just terrible.", 0),

("The film tries to be emotional but ends up being cheesy and fake. The acting is stiff, the writing is cringe, and the pacing is painfully slow. I couldn’t wait for it to end.", 0),

("This movie is so boring it should be illegal. Every scene drags on forever. The characters have no depth, the story has no direction, and the dialogue has no life. Total snoozefest.", 0),

("The directing is clueless. Scenes feel random, the pacing is broken, and the emotional moments fall completely flat. The acting doesn’t help either. Everyone looks bored.", 0),

("The movie feels like a student project that somehow got released. The acting is wooden, the story is predictable, and the emotional scenes feel fake. Just awful.", 0),

("I honestly don’t know how a movie can be this bad. The writing is lazy, the acting is stiff, and the pacing is unbearable. I regret watching it. Easily one of the worst films I’ve seen.", 0),
("I didn’t expect much from this movie, but it honestly warmed my heart. The characters feel real, the dialogue is natural, and the emotional moments hit harder than I expected. The pacing is slow in a comforting way. By the end I felt genuinely attached to everyone. A small but beautiful film.", 1),

("This movie has such a gentle vibe. Nothing feels rushed or forced. The acting is incredibly natural, and the story unfolds with quiet confidence. The emotional payoff in the last act is subtle but powerful. I walked away smiling.", 1),

("The film surprised me with how honest it felt. The performances are grounded, the writing is thoughtful, and the directing is gentle but effective. The emotional scenes land perfectly without being melodramatic. A lovely experience.", 1),

("I loved how the movie lets its characters breathe. No rushed scenes, no unnecessary drama. Just honest storytelling. The chemistry between the leads is fantastic. The ending is bittersweet in the best way.", 1),

("This film is full of small, beautiful moments. The acting is fantastic, the writing is sharp, and the pacing is perfect for the story. I didn’t expect to enjoy it this much, but it really stuck with me.", 1),

("The movie has such a comforting tone. Even when things get heavy, it never feels hopeless. The characters are written with empathy, and the performances are incredibly natural. A genuinely touching film.", 1),

("The directing is subtle but confident. The actors deliver incredibly grounded performances, and the story unfolds with quiet grace. The emotional beats feel earned, not forced. A beautiful little film.", 1),

("This movie is such a pleasant surprise. The pacing is slow but intentional, the acting is fantastic, and the emotional moments feel incredibly real. I walked away feeling genuinely moved.", 1),

("The film balances humor and sadness so well. Nothing feels exaggerated or manipulative. The performances are warm and believable, and the story unfolds naturally. A genuinely heartfelt movie.", 1),

("I didn’t expect to get emotional, but the movie got me good. The writing is thoughtful, the pacing is patient, and the performances are incredibly grounded. The final act is especially strong. Loved it.", 1),

("The chemistry between the leads is incredible. Every scene feels alive and honest. The directing is subtle but effective, and the emotional payoff is fantastic. A genuinely touching film.", 1),

("The film handles emotion with such care. Nothing feels exaggerated or manipulative. The performances are grounded, and the writing is thoughtful. The ending hit me harder than I expected. Beautiful movie.", 1),

("This movie is full of warmth and humanity. The characters feel real, the dialogue is natural, and the emotional moments land perfectly. I didn’t expect to enjoy it this much.", 1),

("The pacing is slow but perfect for the story. The acting is incredibly natural, and the emotional payoff is fantastic. A small but beautiful film.", 1),

("The film has a quiet confidence that I really appreciated. The acting is phenomenal, the writing is sharp, and the emotional beats feel earned. A genuinely touching experience.", 1),

("This movie is full of soft, emotional moments that feel incredibly real. The performances are subtle but powerful, and the writing is thoughtful. The ending is bittersweet and beautiful.", 1),

("The story is simple but told with so much heart. The performances are warm and believable, and the directing is subtle but effective. The final scene is absolutely beautiful. Loved it.", 1),

("The film has such a gentle, melancholic tone. The acting is fantastic, the writing is thoughtful, and the emotional moments feel incredibly honest. A lovely surprise.", 1),

("I didn’t expect to enjoy this as much as I did. The characters are incredibly well-written, the pacing is patient, and the emotional payoff is fantastic. A genuinely heartfelt movie.", 1),

("The movie feels incredibly human. The characters are flawed but lovable, and the story treats them with compassion. The emotional moments feel earned. A beautiful little film.", 1),

("The acting is the standout here. Every expression, every pause feels meaningful. The story is simple but told with so much care. I walked away feeling strangely uplifted.", 1),

("This film surprised me with how emotionally rich it is. The writing is sharp, the characters feel lived-in, and the direction is confident without being flashy. A quiet gem.", 1),

("The movie builds emotion slowly, and when it finally hits, it hits hard. The performances are honest and vulnerable. The writing avoids clichés and lets the characters make believable mistakes.", 1),

("The film balances humor and sadness beautifully. The characters feel real, the dialogue is natural, and the emotional moments land perfectly. A genuinely touching experience.", 1),

("A genuinely heartfelt movie with fantastic performances and thoughtful writing. The emotional moments feel honest and earned. Highly recommended.", 1),
("I honestly don’t understand how a movie can be this dull. Every scene feels like it lasts five minutes too long, and the characters have the emotional depth of cardboard. The plot goes nowhere, and the ending is laughably bad. I kept hoping it would improve, but it never did.", 0),

("This movie is a complete mess from start to finish. The pacing is awful, the acting is stiff, and the writing feels like it was thrown together at the last minute. The emotional scenes are so forced they become unintentionally funny. I regret watching it.", 0),

("The film tries to be deep and meaningful, but it’s just boring and confusing. Characters talk in vague metaphors that don’t mean anything. The pacing is painfully slow. I kept zoning out and didn’t miss a thing. Just bad.", 0),

("I don’t know what the director was going for, but it didn’t work. The tone is inconsistent, the acting is wooden, and the story is full of holes. The emotional moments fall completely flat. Just a frustrating experience.", 0),

("This was honestly painful to sit through. The pacing is glacial, the characters are bland, and the plot is basically nonexistent. The movie keeps pretending something important is happening, but nothing ever does. Total waste of time.", 0),

("The editing is so bad it’s distracting. Scenes cut abruptly, characters appear and disappear randomly, and the pacing is completely broken. The acting doesn’t help either. Everyone looks like they want to go home. Terrible experience.", 0),

("This movie is a perfect example of wasted potential. The idea could’ve worked, but the execution is awful. The writing is lazy, the acting is flat, and the directing is clueless. I was bored the entire time.", 0),

("I swear the movie was written in one night. The dialogue is unnatural, the characters are inconsistent, and the plot is full of holes. The emotional scenes are laughably bad. I couldn’t take any of it seriously.", 0),

("The film is so slow and dull that I almost fell asleep. The characters are boring, the dialogue is repetitive, and the story goes nowhere. The ending tries to be profound but just feels silly. Miserable watch.", 0),

("This movie is just noise. Loud music, loud acting, loud everything. But none of it means anything. The story is paper-thin and the characters are walking clichés. I was annoyed the entire time.", 0),

("The acting is unbelievably bad. Everyone sounds like they’re reading lines off a cue card. The emotional scenes are so overacted they become unintentionally funny. I couldn’t stop cringing.", 0),

("The movie tries to be artistic but ends up being pretentious nonsense. Long shots of nothing happening, characters talking in vague metaphors, and a plot that goes nowhere. I hated every minute.", 0),

("This film is a mess. The pacing is awful, the writing is lazy, and the acting is stiff. The emotional moments feel fake and the humor doesn’t land. I wanted to turn it off halfway through.", 0),

("I don’t know how they managed to make something this boring. The story is predictable, the characters are bland, and the emotional moments feel forced. I couldn’t wait for it to end.", 0),

("The cinematography looks cheap, the lighting is awful, and the sound mix is inconsistent. The story is confusing and the acting is wooden. Nothing about this movie works. Just bad.", 0),

("This movie is so slow it feels like time stops. The characters are uninteresting, the dialogue is flat, and the plot is nonexistent. The ending tries to be emotional but completely fails.", 0),

("The writing is unbelievably lazy. Characters make dumb decisions just to move the plot forward. The emotional scenes are laughably overacted. I couldn’t take any of it seriously.", 0),

("The film feels unfinished. Scenes drag on forever, the pacing is awful, and the acting is stiff. The story goes nowhere and the ending is ridiculous. I regret watching this.", 0),

("This movie is a complete waste of time. The acting is terrible, the writing is worse, and the pacing is unbearable. I kept hoping it would get better, but it only got worse.", 0),

("The plot is full of holes, the pacing is awful, and the acting is shockingly bad. The movie keeps pretending something important is happening, but nothing ever does. Just terrible.", 0),

("The film tries to be emotional but ends up being cheesy and fake. The acting is stiff, the writing is cringe, and the pacing is painfully slow. I couldn’t wait for it to end.", 0),

("This movie is so boring it should be illegal. Every scene drags on forever. The characters have no depth, the story has no direction, and the dialogue has no life. Total snoozefest.", 0),

("The directing is clueless. Scenes feel random, the pacing is broken, and the emotional moments fall completely flat. The acting doesn’t help either. Everyone looks bored.", 0),

("The movie feels like a student project that somehow got released. The acting is wooden, the story is predictable, and the emotional scenes feel fake. Just awful.", 0),

("I honestly don’t know how a movie can be this bad. The writing is lazy, the acting is stiff, and the pacing is unbearable. I regret watching it. Easily one of the worst films I’ve seen.", 0),
("I didn’t expect this movie to hit me the way it did. The characters feel so real and flawed in a way that makes them incredibly easy to care about. The pacing is slow but intentional, and the emotional moments land perfectly. By the end I was genuinely moved. A beautiful little film.", 1),

("This movie has such a warm, human energy. The acting is incredibly natural, and the story unfolds with a quiet confidence. Nothing feels forced or exaggerated. The ending is bittersweet in the best possible way. Loved it.", 1),

("The film surprised me with how honest and grounded it felt. The performances are subtle but powerful, and the writing is thoughtful without being pretentious. The emotional beats feel earned. A genuinely touching experience.", 1),

("I loved how the movie lets its characters breathe. No rushed scenes, no unnecessary drama. Just honest storytelling. The chemistry between the leads is fantastic, and the final act is incredibly moving.", 1),

("This film is full of small, beautiful moments that feel incredibly real. The acting is fantastic, the writing is sharp, and the pacing is perfect for the story. I didn’t expect to enjoy it this much.", 1),

("The movie has such a comforting tone. Even when things get heavy, it never feels hopeless. The characters are written with empathy, and the performances are incredibly natural. A genuinely heartfelt film.", 1),

("The directing is subtle but confident. The actors deliver incredibly grounded performances, and the story unfolds with quiet grace. The emotional beats feel earned, not forced. A beautiful little movie.", 1),

("This movie is such a pleasant surprise. The pacing is slow but intentional, the acting is fantastic, and the emotional moments feel incredibly real. I walked away feeling genuinely moved.", 1),

("The film balances humor and sadness so well. Nothing feels exaggerated or manipulative. The performances are warm and believable, and the story unfolds naturally. A genuinely heartfelt movie.", 1),

("I didn’t expect to get emotional, but the movie got me good. The writing is thoughtful, the pacing is patient, and the performances are incredibly grounded. The final act is especially strong. Loved it.", 1),

("The chemistry between the leads is incredible. Every scene feels alive and honest. The directing is subtle but effective, and the emotional payoff is fantastic. A genuinely touching film.", 1),

("The film handles emotion with such care. Nothing feels exaggerated or manipulative. The performances are grounded, and the writing is thoughtful. The ending hit me harder than I expected. Beautiful movie.", 1),

("This movie is full of warmth and humanity. The characters feel real, the dialogue is natural, and the emotional moments land perfectly. I didn’t expect to enjoy it this much.", 1),

("The pacing is slow but perfect for the story. The acting is incredibly natural, and the emotional payoff is fantastic. A small but beautiful film.", 1),

("The film has a quiet confidence that I really appreciated. The acting is phenomenal, the writing is sharp, and the emotional beats feel earned. A genuinely touching experience.", 1),

("This movie is full of soft, emotional moments that feel incredibly real. The performances are subtle but powerful, and the writing is thoughtful. The ending is bittersweet and beautiful.", 1),

("The story is simple but told with so much heart. The performances are warm and believable, and the directing is subtle but effective. The final scene is absolutely beautiful. Loved it.", 1),

("The film has such a gentle, melancholic tone. The acting is fantastic, the writing is thoughtful, and the emotional moments feel incredibly honest. A lovely surprise.", 1),

("I didn’t expect to enjoy this as much as I did. The characters are incredibly well-written, the pacing is patient, and the emotional payoff is fantastic. A genuinely heartfelt movie.", 1),

("The movie feels incredibly human. The characters are flawed but lovable, and the story treats them with compassion. The emotional moments feel earned. A beautiful little film.", 1),

("The acting is the standout here. Every expression, every pause feels meaningful. The story is simple but told with so much care. I walked away feeling strangely uplifted.", 1),

("This film surprised me with how emotionally rich it is. The writing is sharp, the characters feel lived-in, and the direction is confident without being flashy. A quiet gem.", 1),

("The movie builds emotion slowly, and when it finally hits, it hits hard. The performances are honest and vulnerable. The writing avoids clichés and lets the characters make believable mistakes.", 1),

("The film balances humor and sadness beautifully. The characters feel real, the dialogue is natural, and the emotional moments land perfectly. A genuinely touching experience.", 1),

("A genuinely heartfelt movie with fantastic performances and thoughtful writing. The emotional moments feel honest and earned. Highly recommended.", 1),
("I honestly don’t understand how a movie can be this dull. Every scene drags on forever, the characters have no depth, and the plot feels like it was written in five minutes. The emotional moments are so forced they become unintentionally funny. I kept checking the time, hoping it would end soon.", 0),

("This movie is a complete disaster. The pacing is awful, the acting is stiff, and the writing feels like a first draft someone forgot to rewrite. The emotional scenes fall completely flat. I regret watching it.", 0),

("The film tries so hard to be deep, but it’s just boring and confusing. Characters talk in vague metaphors that don’t mean anything. The pacing is painfully slow. I zoned out multiple times and didn’t miss a thing.", 0),

("I don’t know what the director was going for, but it didn’t work. The tone is inconsistent, the acting is wooden, and the story is full of holes. The emotional moments feel fake. Just a frustrating experience.", 0),

("This was honestly painful to sit through. The pacing is glacial, the characters are bland, and the plot is basically nonexistent. The movie keeps pretending something important is happening, but nothing ever does. Total waste of time.", 0),

("The editing is so bad it’s distracting. Scenes cut abruptly, characters appear and disappear randomly, and the pacing is completely broken. The acting doesn’t help either. Everyone looks bored. Terrible experience.", 0),

("This movie is a perfect example of wasted potential. The idea could’ve worked, but the execution is awful. The writing is lazy, the acting is flat, and the directing is clueless. I was bored the entire time.", 0),

("I swear the movie was written in one night. The dialogue is unnatural, the characters are inconsistent, and the plot is full of holes. The emotional scenes are laughably bad. I couldn’t take any of it seriously.", 0),

("The film is so slow and dull that I almost fell asleep. The characters are boring, the dialogue is repetitive, and the story goes nowhere. The ending tries to be profound but just feels silly. Miserable watch.", 0),

("This movie is just noise. Loud music, loud acting, loud everything. But none of it means anything. The story is paper-thin and the characters are walking clichés. I was annoyed the entire time.", 0),

("The acting is unbelievably bad. Everyone sounds like they’re reading lines off a cue card. The emotional scenes are so overacted they become unintentionally funny. I couldn’t stop cringing.", 0),

("The movie tries to be artistic but ends up being pretentious nonsense. Long shots of nothing happening, characters talking in vague metaphors, and a plot that goes nowhere. I hated every minute.", 0),

("This film is a mess. The pacing is awful, the writing is lazy, and the acting is stiff. The emotional moments feel fake and the humor doesn’t land. I wanted to turn it off halfway through.", 0),

("I don’t know how they managed to make something this boring. The story is predictable, the characters are bland, and the emotional moments feel forced. I couldn’t wait for it to end.", 0),

("The cinematography looks cheap, the lighting is awful, and the sound mix is inconsistent. The story is confusing and the acting is wooden. Nothing about this movie works. Just bad.", 0),

("This movie is so slow it feels like time stops. The characters are uninteresting, the dialogue is flat, and the plot is nonexistent. The ending tries to be emotional but completely fails.", 0),

("The writing is unbelievably lazy. Characters make dumb decisions just to move the plot forward. The emotional scenes are laughably overacted. I couldn’t take any of it seriously.", 0),

("The film feels unfinished. Scenes drag on forever, the pacing is awful, and the acting is stiff. The story goes nowhere and the ending is ridiculous. I regret watching this.", 0),

("This movie is a complete waste of time. The acting is terrible, the writing is worse, and the pacing is unbearable. I kept hoping it would get better, but it only got worse.", 0),

("The plot is full of holes, the pacing is awful, and the acting is shockingly bad. The movie keeps pretending something important is happening, but nothing ever does. Just terrible.", 0),

("The film tries to be emotional but ends up being cheesy and fake. The acting is stiff, the writing is cringe, and the pacing is painfully slow. I couldn’t wait for it to end.", 0),

("This movie is so boring it should be illegal. Every scene drags on forever. The characters have no depth, the story has no direction, and the dialogue has no life. Total snoozefest.", 0),

("The directing is clueless. Scenes feel random, the pacing is broken, and the emotional moments fall completely flat. The acting doesn’t help either. Everyone looks bored.", 0),

("The movie feels like a student project that somehow got released. The acting is wooden, the story is predictable, and the emotional scenes feel fake. Just awful.", 0),

("I honestly don’t know how a movie can be this bad. The writing is lazy, the acting is stiff, and the pacing is unbearable. I regret watching it. Easily one of the worst films I’ve seen.", 0),


]

# Wklej tutaj wynik PROMPT_20NG_200
newsgroups_augmented = [
   ("Honestly I’m tired of people acting like morality collapses without religion. Plenty of us manage just fine without a holy rulebook.", "alt.atheism"),
("Can someone explain why every debate ends with 'you just need faith'? That’s not an argument, it’s an escape hatch.", "alt.atheism"),
("My coworker told me atheists 'secretly believe'. Dude, I barely believe my alarm clock will ring on time.", "alt.atheism"),
("Why do people assume atheism = anger? I’m not angry, I’m just unconvinced. Big difference.", "alt.atheism"),
("Someone in the thread said atheists have no purpose. Bro, I have laundry, bills, and a cat. That’s purpose enough.", "alt.atheism"),
("I tried discussing evidence with a believer today. Five minutes in, they switched to personal attacks. Classic.", "alt.atheism"),
("Funny how people say atheists are 'closed‑minded' while refusing to question anything themselves.", "alt.atheism"),
("Anyone else get the 'you’ll believe when you’re older' speech? I’m 34. Still waiting.", "alt.atheism"),
("I don’t hate religion, I just don’t buy it. Why is that so hard for some folks to accept?", "alt.atheism"),
("The thread about miracles is wild. Half the stories sound like coincidences and the other half like bad memory.", "alt.atheism"),

("Anyone know why my OpenGL textures look super blurry when rotating? I swear I set the filtering right.", "comp.graphics"),
("Trying to export a model from Blender and the normals keep flipping. Is this a known bug or am I cursed?", "comp.graphics"),
("Does anyone have a simple explanation for UV unwrapping that doesn’t sound like a math lecture? I’m lost.", "comp.graphics"),
("My GPU crashes every time I try to render anything above 4K. Temps look fine. Any ideas?", "comp.graphics"),
("Why do all tutorials assume you already know half the pipeline? I just want to make a cube look shiny.", "comp.graphics"),
("Anyone here still using POV-Ray? I found some old files and now I’m nostalgic as hell.", "comp.graphics"),
("Is there a lightweight tool for converting .obj to .fbx without messing up materials? Preferably free.", "comp.graphics"),
("I tried learning shaders again and my brain melted. How do people make this look easy?", "comp.graphics"),
("My animation looks smooth in viewport but choppy in final render. Is this a frame rate thing or something else?", "comp.graphics"),
("Anyone got tips for reducing noise in Cycles without waiting 20 minutes per frame? My PC is crying.", "comp.graphics"),

("Windows keeps randomly minimizing all my windows like it’s having a panic attack. Anyone know what causes this?", "comp.os.ms-windows.misc"),
("Why does the system freeze every time I plug in a USB drive? Device Manager shows nothing weird. I’m losing my mind.", "comp.os.ms-windows.misc"),
("Is there a way to stop Windows from rebooting itself for updates at the worst possible moment? It’s like it hates productivity.", "comp.os.ms-windows.misc"),
("My taskbar icons keep disappearing after sleep mode. Not all of them, just random ones. Any fix for this nonsense?", "comp.os.ms-windows.misc"),
("Anyone else getting that weird 'This app can’t run on your PC' error for apps that ran fine yesterday?", "comp.os.ms-windows.misc"),
("Windows Search is completely broken on my machine. It finds files from 2012 but not the one I saved 10 minutes ago.", "comp.os.ms-windows.misc"),
("Why does File Explorer take 5 seconds to open a folder with like 12 files? This shouldn’t be that hard.", "comp.os.ms-windows.misc"),
("My audio randomly switches to some phantom device that doesn’t exist. I have to restart to fix it. Any ideas?", "comp.os.ms-windows.misc"),
("Windows Defender keeps flagging my own scripts as malware. They’re literally just batch files. How do I whitelist them properly?", "comp.os.ms-windows.misc"),
("Anyone know why Windows Update gets stuck at 0% forever? I’ve tried restarting, resetting, sacrificing a goat… nothing works.", "comp.os.ms-windows.misc"),

("My old IBM-compatible tower keeps randomly shutting down under load. PSU is new, temps look fine. Could it be the motherboard caps finally dying?", "comp.sys.ibm.pc.hardware"),
("Anyone know why my BIOS refuses to detect a second IDE drive? Jumpers are correct, cables swapped, still nothing.", "comp.sys.ibm.pc.hardware"),
("Trying to upgrade RAM on an ancient Pentium board and it only boots with one stick. Both sticks work alone. Is this a compatibility thing?", "comp.sys.ibm.pc.hardware"),
("My floppy drive keeps clicking nonstop even when idle. Is it dying or is this some weird controller issue?", "comp.sys.ibm.pc.hardware"),
("Does anyone remember how to disable onboard video on these old IBM clones? I installed a PCI GPU and now both outputs are dead.", "comp.sys.ibm.pc.hardware"),
("I swapped the CMOS battery but the system still forgets the date every reboot. Could the RTC chip itself be failing?", "comp.sys.ibm.pc.hardware"),
("Trying to install a larger IDE HDD and the BIOS refuses to recognize anything above 8GB. Is there a workaround without flashing a modded BIOS?", "comp.sys.ibm.pc.hardware"),
("My keyboard randomly stops responding until I wiggle the connector. Is this a port issue or just a dying cable?", "comp.sys.ibm.pc.hardware"),
("Anyone know why my ISA sound card causes the whole system to freeze when playing audio? IRQ conflict maybe?", "comp.sys.ibm.pc.hardware"),
("The machine boots but gives no video signal. Fans spin, drives spin, but nothing on screen. Already tried another GPU. Could it be RAM or CPU?", "comp.sys.ibm.pc.hardware"),

("My old MacBook Pro keeps randomly shutting down at 40–50 battery. CoconutBattery says the health is fine. Is this a known issue?", "comp.sys.mac.hardware"),
("Anyone know why my iMac fans ramp to max even when I’m just browsing? Temps look normal in iStat.", "comp.sys.mac.hardware"),
("Upgraded RAM in my 2012 Mac Mini and now it only boots every second try. Both sticks work alone. Compatibility problem?", "comp.sys.mac.hardware"),
("My MacBook keyboard is double‑typing certain letters. I cleaned it, reset SMC, nothing helps. Is the butterfly keyboard really this bad?", "comp.sys.mac.hardware"),
("External monitor flickers like crazy when connected via USB‑C. Tried different cables. Could it be the adapter?", "comp.sys.mac.hardware"),
("Anyone else have issues with the trackpad randomly freezing for a second or two? Happens a few times a day, super annoying.", "comp.sys.mac.hardware"),
("My Mac Pro refuses to recognize a new SSD unless I reboot twice. Disk Utility sees it sometimes, sometimes not. Weird hardware glitch?", "comp.sys.mac.hardware"),
("The speakers on my MacBook Air started crackling out of nowhere. No drops, no water. Is this a known failure?", "comp.sys.mac.hardware"),
("Trying to replace the thermal paste on a 2015 MBP and the screws are insanely tight. Any tips before I strip them?", "comp.sys.mac.hardware"),
("My Time Machine drive disconnects randomly even though it’s plugged directly into the Mac. Cable issue or port dying?", "comp.sys.mac.hardware"),

("Anyone know why my X server randomly refuses to start after an update? It just dumps me back to the console with no error.", "comp.windows.x"),
("Trying to configure dual monitors in X and it keeps mirroring no matter what I put in xrandr. Am I missing something obvious?", "comp.windows.x"),
("My window manager crashes every time I switch workspaces quickly. Logs don’t show anything useful. Any ideas?", "comp.windows.x"),
("Is there a simple guide for setting up a custom xinitrc? Everything online assumes you already know half the system.", "comp.windows.x"),
("X11 forwarding over SSH is painfully slow on my machine. Even simple apps lag like crazy. Is this normal?", "comp.windows.x"),
("Anyone else having issues with fonts looking blurry in X but fine in Wayland? I swear I didn’t change anything.", "comp.windows.x"),
("My mouse acceleration in X is completely unpredictable. Sometimes it’s smooth, sometimes it jumps like crazy. How do people deal with this?", "comp.windows.x"),
("Trying to get a tablet working under X and the stylus maps to the wrong screen every time. xinput isn’t helping.", "comp.windows.x"),
("Why does configuring keybindings in X feel like solving a puzzle from another dimension? Nothing is where you expect it.", "comp.windows.x"),
("Xorg keeps ignoring my custom config in /etc/X11/xorg.conf.d. It loads defaults instead. Permissions look fine. What else should I check?", "comp.windows.x"),

("My bike started making a weird rattling noise around 4k RPM. Chain looks fine, exhaust tight. Any ideas what else to check?", "rec.motorcycles"),
("Thinking about getting my first bike. Is a used Ninja 300 a decent starter or should I go even smaller?", "rec.motorcycles"),
("Anyone know why my clutch lever feels spongy after a long ride? Fluid level is okay. Maybe air in the line?", "rec.motorcycles"),
("Dropped my bike at a stop sign today… only scratched the fairing but my ego is destroyed. Happens to everyone, right?", "rec.motorcycles"),
("My rear brake squeals like crazy even after cleaning the rotor. Pads still have life. What else could cause it?", "rec.motorcycles"),
("Looking for recommendations on a good budget helmet. Something safe but not $500. Any favorites?", "rec.motorcycles"),
("Bike keeps stalling when cold unless I give it throttle. Carb issue? Idle screw? Not sure where to start.", "rec.motorcycles"),
("Anyone here ride year‑round? How do you deal with fogging visors in winter? Mine fogs instantly.", "rec.motorcycles"),
("Thinking about switching to synthetic oil. Worth it on an older engine or just marketing hype?", "rec.motorcycles"),
("My turn signals blink super fast after replacing bulbs. Do I need resistors or a different relay?", "rec.motorcycles"),

("My car started shaking around 70 mph and I already had the wheels balanced last week. Could this be a bent rim?", "rec.autos"),
("Check engine light keeps coming back every time I clear it. Car drives fine but the code returns after 10–15 miles. Any ideas?", "rec.autos"),
("Thinking about buying a used 2008 Civic. Anything specific I should look out for when checking it?", "rec.autos"),
("My engine makes a loud ticking noise when cold but quiets down after warming up. Lifters or something worse?", "rec.autos"),
("Anyone here installed a backup camera themselves? Wondering if it’s doable without tearing half the interior apart.", "rec.autos"),
("My car burns about a quart of oil every 1500 miles. Mechanic says 'these engines do that'. Is that actually true?", "rec.autos"),
("Looking for a good fuel injector cleaner. Car hesitates at low RPM and I’m hoping it’s something simple.", "rec.autos"),
("Temperature gauge jumps up a few bars when I’m stuck in traffic. Fan works. Could this be a thermostat issue?", "rec.autos"),
("Are all‑season tires worth it if I only drive 6–8k miles a year? Opinions online are all over the place.", "rec.autos"),
("After replacing the battery my radio wants a code I don’t have. Is there any way to recover it without going to the dealer?", "rec.autos"),

("Selling a 19'' LCD monitor. Works fine but has a slight discoloration in the corner. Local pickup only, it's heavy.", "misc.forsale"),
("Giving away an old wooden desk for $5. Scratched but sturdy. First come, first served.", "misc.forsale"),
("Looking to buy a cheap used GPU, something like a GTX 750 Ti. Doesn’t need to look pretty, just needs to run.", "misc.forsale"),
("Selling a city bike, decent condition, brakes squeal a bit. Price negotiable.", "misc.forsale"),
("Free laser printer. I think it works but I don’t have toner to test it. Take it if you want it.", "misc.forsale"),
("Buying a budget laptop for basic tasks. Doesn’t need to be fast, just needs a battery that lasts more than 10 minutes.", "misc.forsale"),
("Selling a 2.1 speaker set. Sounds okay but the subwoofer hums sometimes. $20 and it’s yours.", "misc.forsale"),
("Got a bunch of HDMI and USB cables plus some old power adapters. Selling cheap if anyone needs them.", "misc.forsale"),
("Looking for a used 24'' monitor with no dead pixels. Older model is fine as long as the picture is sharp.", "misc.forsale"),
("Selling an office chair. Comfortable, but one armrest is a bit loose. Low price because I don’t feel like fixing it.", "misc.forsale"),

("I swear the bullpen is going to give me a heart attack this season. How do you blow a lead three games in a row?", "rec.sport.baseball"),
("Anyone else think the new pitch clock actually makes the game feel smoother? I didn’t expect to like it but I kinda do.", "rec.sport.baseball"),
("My team keeps leaving runners on base like it’s a hobby. How hard is it to get a single with men on second and third?", "rec.sport.baseball"),
("Does anyone have the updated spring training schedule? The website keeps giving me a 404.", "rec.sport.baseball"),
("I still don’t understand why they traded away their best hitter for 'future prospects'. We’ve been rebuilding for a decade.", "rec.sport.baseball"),
("The umpiring this week has been awful. Strike zone looks like it’s being drawn by a drunk guy with a crayon.", "rec.sport.baseball"),
("Anyone here actually like the new uniforms? They look like cheap knockoffs you’d find at a gas station.", "rec.sport.baseball"),
("My kid wants to start Little League this year. Any recommendations for a decent beginner glove that won’t fall apart?", "rec.sport.baseball"),
("I miss the old stadium. The new one is nice but it feels too corporate. No soul, just ads everywhere.", "rec.sport.baseball"),
("Why do commentators act like bunting is some ancient forbidden art? Sometimes it’s literally the smartest play.", "rec.sport.baseball"),

("Our defense has been a disaster this season. You can’t win games when you leave your goalie out to dry every night.", "rec.sport.hockey"),
("Anyone else think the new goalie pads look ridiculous? They’re huge but somehow he still lets everything in.", "rec.sport.hockey"),
("I miss the old physical style of hockey. Everything now feels softer and over‑regulated.", "rec.sport.hockey"),
("Does anyone have a stream link for tonight’s game? The official site keeps buffering like it’s 1999 dial‑up.", "rec.sport.hockey"),
("How do you blow a 3‑goal lead in the third period? This team finds new ways to disappoint me.", "rec.sport.hockey"),
("My kid wants to start playing hockey. Any recommendations for beginner skates that won’t cost a fortune?", "rec.sport.hockey"),
("The refs last night were unbelievable. Missed calls everywhere and then they call a phantom penalty in OT.", "rec.sport.hockey"),
("Why does our power play look like they’ve never practiced together? Zero movement, zero creativity.", "rec.sport.hockey"),
("Anyone else think the new arena feels too corporate? Great seats, but no atmosphere at all.", "rec.sport.hockey"),
("I still can’t believe they traded our best winger for 'future picks'. We’ve been rebuilding forever.", "rec.sport.hockey"),

("I can’t believe they left the starter in for the 8th inning. He was clearly gassed and everyone saw it coming.", "rec.sport.baseball"),
("Anyone else think the new analytics‑driven lineups are overcomplicating things? Sometimes you just need your best hitters hitting.", "rec.sport.baseball"),
("The broadcast team last night was unbearable. They spent half the game talking about hot dogs instead of the actual plays.", "rec.sport.baseball"),
("Why does our closer only pitch in save situations? He’s our best arm, use him when the game is on the line.", "rec.sport.baseball"),
("I swear the outfield defense has regressed. Routine fly balls suddenly look like adventures.", "rec.sport.baseball"),
("Does anyone know if the minor league prospect list got updated? I heard two of our guys jumped into the top 50.", "rec.sport.baseball"),
("The new stadium food prices are insane. $14 for a pretzel? Come on.", "rec.sport.baseball"),
("Our catcher can’t frame a pitch to save his life. Every borderline strike turns into a ball.", "rec.sport.baseball"),
("Anyone else think the replay system is getting out of hand? Takes forever and half the time they still get it wrong.", "rec.sport.baseball"),
("My kid wants to start collecting baseball cards. Any tips on where to buy packs that aren’t overpriced?", "rec.sport.baseball"),

("Does anyone know if the new hash function proposal has any known collision attacks yet? I can’t find anything recent.", "sci.crypt"),
("I’m trying to understand why people still use MD5 for anything. It’s been broken for years and yet it keeps showing up.", "sci.crypt"),
("Is there a simple explanation for why RSA key generation takes so long on older machines? Mine feels like it’s mining gold.", "sci.crypt"),
("Someone at work suggested rolling our own encryption. I nearly fainted. How do people still think that’s a good idea?", "sci.crypt"),
("Does anyone have a good resource explaining side‑channel attacks in a non‑academic way? Most papers are unreadable.", "sci.crypt"),
("I’m trying to implement AES in Python and keep getting different outputs from OpenSSL. Probably a padding issue but I’m stuck.", "sci.crypt"),
("Why do people keep confusing hashing with encryption? They’re not even remotely the same thing.", "sci.crypt"),
("Is PGP basically dead at this point? Feels like nobody uses it anymore except old mailing lists.", "sci.crypt"),
("Anyone know if quantum‑resistant algorithms are actually practical yet or still mostly theoretical hype?", "sci.crypt"),
("Trying to explain public‑key crypto to a friend and realized I barely understand it myself. Any good beginner‑friendly guides?", "sci.crypt"),

("I’m trying to build a simple linear power supply and the regulator keeps overheating even with a heatsink. Do I just need a bigger one?", "sci.electronics"),
("Anyone know why my breadboard circuit works fine until I touch it? Feels like some grounding or noise issue but I’m not sure.", "sci.electronics"),
("Looking for a good explanation of why switching power supplies are so noisy. Every article I find is way too academic.", "sci.electronics"),
("My multimeter shows random voltage spikes when measuring a DC line. Bad meter, bad cable, or something actually wrong?", "sci.electronics"),
("Trying to repair an old radio and the electrolytic caps look suspicious. Should I just replace all of them or only the obviously bad ones?", "sci.electronics"),
("Does anyone have experience with cheap oscilloscope clones? Worth it for hobby work or just garbage?", "sci.electronics"),
("I accidentally reversed polarity on a small DC motor and now it barely spins. Did I fry it or is there something to salvage?", "sci.electronics"),
("Why do some LEDs glow faintly even when the circuit is off? Is this leakage current or something else?", "sci.electronics"),
("I’m trying to understand transistor biasing and every tutorial uses different notation. Is there a standard or is it chaos?", "sci.electronics"),
("Anyone know a good beginner‑friendly book on analog electronics? Everything I find jumps straight into heavy math.", "sci.electronics"),

("Is it normal for a cold to last more than two weeks? Symptoms aren’t getting worse but they’re not going away either.", "sci.med"),
("Anyone know if long‑term use of ibuprofen can mess with your stomach? I’ve been taking it daily for back pain.", "sci.med"),
("I keep waking up with numb hands. Could this be carpal tunnel or something else entirely?", "sci.med"),
("Does anyone have experience with elimination diets? Trying to figure out if certain foods are triggering my migraines.", "sci.med"),
("My doctor says my blood pressure is 'borderline' but didn’t explain what that means. Should I be worried?", "sci.med"),
("Is it true that drinking too much water can actually be dangerous? I thought hydration was always good.", "sci.med"),
("I’ve been getting random heart palpitations lately. Not painful, just weird. Anyone dealt with this before?", "sci.med"),
("Does anyone know if vitamin D supplements actually help with fatigue or if it’s just placebo?", "sci.med"),
("I’m trying to understand the difference between viral and bacterial infections. Why do symptoms feel so similar?", "sci.med"),
("My sleep schedule is completely wrecked. Any science‑based tips for resetting it without medication?", "sci.med"),

("Is it true that the Artemis lander design is still being revised? Hard to keep track with all the delays.", "sci.space"),
("Anyone know why SpaceX keeps changing the Starship heat shield tiles? Are they still having issues with them falling off?", "sci.space"),
("I’ve been trying to understand how ion thrusters actually produce enough thrust to be useful. The numbers seem tiny.", "sci.space"),
("Does anyone have a good explanation of why Mars dust storms can last for months? Seems wild for a planet with such thin air.", "sci.space"),
("Is the Voyager 1 telemetry issue actually fixable or is this the beginning of the end for the mission?", "sci.space"),
("Why don’t we build telescopes on the far side of the Moon? Seems like the perfect radio‑quiet environment.", "sci.space"),
("Anyone else think the new exoplanet discoveries are getting overshadowed by launch news? Some of these planets are insane.", "sci.space"),
("Trying to wrap my head around how gravitational assists work. How does stealing momentum from a planet not slow it down?", "sci.space"),
("Is there any realistic chance of a crewed mission to Mars before 2040 or is that just PR optimism?", "sci.space"),
("Does anyone know if the James Webb mirror alignment process is fully automated or still requires manual tweaking?", "sci.space"),

("Does anyone else struggle with balancing faith and modern life? Some days it feels impossible to keep up with both.", "soc.religion.christian"),
("I’ve been trying to read the Bible daily but keep losing momentum. Any tips for building a consistent habit?", "soc.religion.christian"),
("Is it normal to feel disconnected during church services? I’m there physically but my mind drifts constantly.", "soc.religion.christian"),
("Anyone have recommendations for good introductory theology books? Something not too academic.", "soc.religion.christian"),
("I’m confused about the differences between denominations. Are the doctrinal gaps really that big or mostly historical?", "soc.religion.christian"),
("How do you handle conversations with people who think faith is outdated? I never know what to say.", "soc.religion.christian"),
("Does anyone else feel guilty when they miss prayer time? I know it’s not supposed to be about perfection.", "soc.religion.christian"),
("Looking for a church that’s welcoming but not overly modernized. Hard to find that balance these days.", "soc.religion.christian"),
("Is it okay to question parts of scripture? I’ve always been told doubt is dangerous, but I’m not sure that’s true.", "soc.religion.christian"),
("Anyone here involved in community outreach programs? Thinking about joining one but not sure where to start.", "soc.religion.christian"),

("I still don’t understand how new gun laws are supposed to stop criminals who already ignore the old ones.", "talk.politics.guns"),
("Anyone know if the new background check proposal actually changes anything or is it just political theater?", "talk.politics.guns"),
("People keep arguing about banning certain rifles, but nobody talks about mental health funding. Feels like the real issue.", "talk.politics.guns"),
("Why do politicians who have armed security insist regular people don’t need self‑defense? The hypocrisy is wild.", "talk.politics.guns"),
("Does anyone have stats on whether concealed carry actually reduces crime? I’ve seen claims both ways.", "talk.politics.guns"),
("Every time there’s a high‑profile incident, the same arguments come out. No one listens, everyone just yells.", "talk.politics.guns"),
("I’m curious if any country has successfully reduced gun violence without banning everything. Examples seem rare.", "talk.politics.guns"),
("People keep saying 'just enforce the laws we have', but half of them aren’t enforced at all. Why is that?", "talk.politics.guns"),
("Is there any evidence that mandatory training improves safety, or is it just another barrier to ownership?", "talk.politics.guns"),
("The media always uses the term 'assault weapon' but nobody can define it consistently. It’s basically a vibe at this point.", "talk.politics.guns"),

("It feels like every peace negotiation ends the same way: big promises, zero follow‑through. Hard to stay optimistic.", "talk.politics.mideast"),
("Does anyone have a good source explaining the historical borders in the region? Every map I find tells a different story.", "talk.politics.mideast"),
("People keep acting like the conflict is simple, but the political layers go back decades. Nothing about it is straightforward.", "talk.politics.mideast"),
("I’m trying to understand why foreign powers keep getting involved when they clearly make things worse half the time.", "talk.politics.mideast"),
("Is there any evidence that economic sanctions actually help stabilize the region? Results seem mixed at best.", "talk.politics.mideast"),
("Media coverage is so inconsistent. One outlet focuses on one side, another on the opposite. Hard to get a balanced picture.", "talk.politics.mideast"),
("Anyone know if the latest ceasefire is holding? Last I heard, both sides were accusing each other of violations.", "talk.politics.mideast"),
("It’s frustrating how every political leader claims to want peace but keeps escalating rhetoric whenever convenient.", "talk.politics.mideast"),
("Does anyone have recommendations for books that explain the conflict without pushing a political agenda?", "talk.politics.mideast"),
("Sometimes it feels like ordinary people want peace more than any government involved. The politics just get in the way.", "talk.politics.mideast"),

("Every election cycle feels the same: lots of promises, very little follow‑through once people get into office.", "talk.politics.misc"),
("Anyone else tired of political debates turning into shouting matches? Nobody actually listens anymore.", "talk.politics.misc"),
("I still don’t understand why people vote strictly along party lines. Doesn’t anyone evaluate candidates individually?", "talk.politics.misc"),
("The amount of misinformation online is insane. Hard to tell what’s real without digging through five sources.", "talk.politics.misc"),
("Why do politicians keep talking about 'unity' while doing the exact opposite? The disconnect is wild.", "talk.politics.misc"),
("Does anyone know if the new policy proposal has any real economic analysis behind it or just slogans?", "talk.politics.misc"),
("Feels like political discussions with friends have become impossible. Everyone is on edge all the time.", "talk.politics.misc"),
("I wish more people admitted they don’t fully understand the issues. Everyone pretends to be an expert.", "talk.politics.misc"),
("Is there any country where political ads are actually regulated well? Ours feel like pure chaos.", "talk.politics.misc"),
("Sometimes I think the average voter is more reasonable than the people representing them. The system feels upside down.", "talk.politics.misc"),

]

# How We Taught an AI to Drive a Race Car

A project journal, written for adults, to be adapted into kid-friendly language for
Mikey's school project. It tells the story of what we tried, what went wrong, and what
each failure taught us, from the first crash at Turn 1 to an AI that is learning to lap
the whole track.

Last updated: 2026-06-19.

---

## The mission

This is a father-and-son project. Mike handles the engineering; Mikey, who is nine, is
the product owner, he decides what the car should be rewarded for and what we try next.
The goal is to train an artificial intelligence to drive a race car around a track in a
realistic driving simulator called BeamNG.tech, entirely on its own, by learning rather
than by being told what to do.

The car is a Civetta Scintilla: 748 horsepower, rear-wheel drive. That detail matters
more than anything else in this story. A car with that much power sending it all to the
rear wheels is extremely easy to spin. Press the gas too hard in a corner and the back
end breaks loose. Much of the project has been about teaching the AI the very thing a
human racing driver spends years learning: how to be gentle and precise with that power.

Our test track is "West Coast USA," about 4,356 meters around with 16 corners. The first
corner, Turn 1, is a fast left-hander roughly 300 meters after the start line. Turn 1
became the wall the project lived and died on for months. Almost every version of our AI
failed there, in a different way each time, and almost everything we learned came from
figuring out why.

There is a real deadline: our simulator license expires on August 6, 2026, so everything
has to work before then.

## How an AI learns to drive

The technique is called reinforcement learning. Instead of programming the rules of
driving, we let the AI try, give it a score for how well it did, and let it adjust itself
to earn a higher score next time. Over millions of attempts it gradually discovers what
works. Nobody tells it "brake before the corner." If braking earns more points than not
braking, it figures that out by itself.

Three pieces make this work, and almost every problem we hit traced back to one of them:

The first is the reward, the scoring system. This is the single most important and most
dangerous thing in the whole project. The AI will do whatever earns the most points, even
if that is not at all what we meant. Designing the reward is Mikey's job, and it turns out
to be genuinely hard.

The second is the observation, what the AI can "see" each instant. Our AI sees about 18
numbers many times a second: its speed, how far it has drifted from the ideal path, how
sharp the upcoming corner is, whether its back end is sliding, and so on. If it cannot see
a corner coming, it cannot learn to slow down for it.

The third is the curriculum, the situations it gets to practice. An AI only learns from
what it experiences. If it always starts from the same spot and crashes at Turn 1, it
never sees Turn 2, so it can never learn Turn 2.

The learning algorithm itself is called SAC, and it runs on a dedicated training computer
we call "the mini." A single training session takes many hours, so we cannot afford to
waste them.

## The story, told through its failures

### The car kept crashing at Turn 1

In the early versions, the car could not get through Turn 1. It would wobble from side to
side down the straight, the wobble would grow with speed, and eventually it would spin.
This left-right weave haunted the project for a long time.

### The trap of telling it what to do

Our first instinct was to stop the spinning by adding rules: traction control to cut the
throttle when the wheels slipped, stability control to cut it when the car slid sideways,
limits on how fast and how far it could steer. Each rule helped a little, but we slowly
realized we were hand-building a robot driver and bolting it onto the AI, not letting the
AI learn anything. Mikey put it best: "we are engineering a scripted robot driver, not
reinforcement learning."

We stopped and did a careful review of how the world's best racing AIs are built. The
answer was clear and humbling: braking and cornering are supposed to *emerge* from the
reward, never to be scripted. Piling on hand-coded rules is a known anti-pattern. The rules
fight the learner and prevent it from ever learning the skill itself. We had been treating
the symptom instead of fixing the cause, which was that our reward and our observations
were wrong.

### The reset: rewarding the right things

So we threw the rules out and rebuilt the foundation. We computed a target speed for every
point on the track, slow for the tight corners, fast for the straights, and rewarded the
car for matching it. We gave the AI better eyes: it could now see how sharp the corner
ahead was, scaled to how fast it was going, and feel when its back end was starting to
slide. The idea was to let the right behavior grow out of good information and good
incentives, instead of forcing it with rules.

### The lazy AI

The reset led straight to one of the best lessons of the project. The AI found a loophole.
It discovered that if it crawled along at walking pace, it never crashed, and it could
safely collect a tiny trickle of points forever. The scoreboard looked like it was
climbing, but the car was barely moving. This is the central truth of reinforcement
learning: the AI does not do what you want, it does what scores. We learned to stop
trusting the score charts and instead watch the actual car, because the charts lied to us
again and again.

The fix was to add a reward for carrying speed, up to the safe limit, so that crawling
stopped being the easy way to win. It worked beautifully. The car went from crawling at
walking speed to driving the straight at about 75 kilometers per hour. That was a real
breakthrough.

### When a fix backfires

Now the car drove fast and even braked for Turn 1, but at the moment it turned in, it
would floor the throttle, the back end would step out, and it would spin off. Classic
too-much-power-in-a-corner. So we turned up a "don't slide" penalty to teach it to be
gentle on the gas.

It backfired in a way that taught us something deep. The car learned that turning hard is
what makes it slide, so to avoid the penalty it simply stopped turning. It drove straight
off the road instead. The problem was that "cornering hard at the limit" and "starting to
spin" look almost identical to the penalty, they both show up as the back end sliding, so
punishing one punished the other. You cannot fix a precision problem with a blunt
instrument.

### The racing line

The real breakthrough was the racing line. Instead of asking the AI to follow the middle
of the road, we used math to compute the ideal path through every corner, the smoothest,
fastest line that swings wide on entry, clips the inside of the corner at the apex, and
swings wide again on exit. We fed that line to the AI as the path to follow.

This was the unlock. For the first time, the car followed the line and actually entered
Turn 1 well. It was the best driving we had ever seen. But two problems remained: the old
left-right wobble, and braking too hard in the middle of the corner, which still made the
back end come around. And we hit a ceiling. Pure trial-and-error learning, on our single
computer, could get the car close to solving Turn 1 but not all the way, and then it would
actually start getting worse. This is not surprising: the most famous racing AI ever built
used more than a thousand computers and the equivalent of decades of driving practice. We
have one mini.

### Training wheels

That brings us to where we are now. We built a simple, hand-coded autopilot that follows
the racing line perfectly and brakes at exactly the right points. For the first time in the
entire project, something drove the whole track, all 4,326 meters, with a textbook-clean
Turn 1. That autopilot is not the AI; it is a scaffold.

The plan is to use it as training wheels. We let the autopilot drive while the AI watches
and learns from it, then we slowly fade the autopilot away so the AI takes over more and
more of the driving, until it is driving entirely on its own with the training wheels gone.
If it works, the end product is a pure AI that learned to drive the track itself, having
been shown what good looks like instead of having to discover it from scratch. There is a
safety net: if the AI cannot quite take over completely on our limited computer, we keep a
thin sliver of the autopilot underneath. Either way, we get a car that laps.

## What we learned

The reward is everything, and the score charts lie. The most repeated lesson by far. Time
and again the score looked great while the car was failing, the lazy crawl that scored
well, the speed reward that paid the car just for moving. The only reliable test was to
watch the actual car drive.

Do not script the behavior you want the AI to learn. Every rule we bolted on to force good
driving got in the way of the AI learning to drive well. The durable fixes were always to
the reward, the observations, and the practice, never another rule.

AIs find loopholes. The crawling car is the perfect example. If there is a cheap way to
score that you did not intend, the AI will find it.

A penalty can teach the wrong lesson. The "don't slide" penalty taught the car to stop
turning, because it could not tell good cornering from a spin.

Practice everything, not just the first problem. As long as the car always died at Turn 1,
it never learned the rest of the track. We had to start it from random points all around
the track so it could practice every corner.

Measure before you change anything. More than once we almost "fixed" something based on a
guess that turned out to be wrong. By actually measuring, we discovered that a supposed
cliff at Turn 1 did not exist, and that a hidden bug was falsely killing the car on certain
stretches of road, a bug that had been lurking for many versions and only showed up once
the autopilot drove far enough to reach those stretches.

Pure trial-and-error has limits. Sometimes the smart move is to give the AI a head start
from something that already works, and then let it learn the genuinely hard part, rather
than insisting it discover everything alone.

## Where we are now, and who did what

The autopilot laps the whole track and Turn 1 is solved. We are now teaching the AI to take
over from the autopilot and eventually drive on its own. The goal we have chased since the
beginning, getting cleanly through Turn 1 and onto the lap, is finally within reach.

The work was shared. Mike built and debugged the machinery and ran the training. Mikey, the
product owner, decided what the car should be rewarded for and what to try next, and watched
the car drive to judge whether it was really working. Two AI assistants helped: one wrote
and tested the code on the training computer, and one acted as a strategy and analysis
partner, reading the results, doing the research, and planning each next step. Every change
we made traces back to a decision one of us made on purpose, which is exactly how we wanted
it.

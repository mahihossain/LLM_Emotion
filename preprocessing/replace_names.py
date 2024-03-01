# script to replace the class names with verbal explanations that can be directly used to generate prompts

import pandas as pd
import pdb


data_path = '../data/2deep_transcript_cleaned.csv' # TODO: Sayed, is this the correct data path?
df = pd.read_csv(data_path)


name_map = {
    'Gaze': {
        'DOWN': 'The interviewee looks down.',
        'DOWN_LEFT': 'The interviewee looks down and left.',
        'DOWN_RIGHT': 'The interviewee looks down and right.',
        'LEFT': 'The interviewee looks to the left.',
        'RIGHT': 'The interviewee looks to the right.',
        'STRAIGHT_AT_INTERVIEWER': 'The interviewee looks straight at the interviewer.',
        'UP': 'The interviewee looks up.',
        'UP_LEFT': 'The interviewee looks up and left.',
        'UP_RIGHT': 'The interviewee looks up and right.'
    },
    'Eyes': {
        'PINCH': 'The interviewee squeezes their eyes together.',
        'BLINK_REPEATEDLY': 'The interviewee blinks more than once in a row.',
        'CLOSE': 'The eyes of the interviewee are closed.',
        'WIDEN': 'The interviewee widens their eyes.'
    },
    'Ekman expression': {
        'FEAR': 'The interviewee shows the Ekman expression fear.',
        'DISGUST': 'The interviewee shows the Ekman expression disgust.',
        'JOY': 'The interviewee shows the Ekman expression joy.',
        'SURPRISE': 'The interviewee shows the Ekman expression surprise.'
    },
    'Smile': {
        'DUCHENNE': 'The interviewee shows a Duchenne smile, i.e. a smile that reaches the eyes.',
        'NON_DUCHENNE': 'The interviewee shows a non-Duchenne smile, i.e. a smile that concentrates only on the mouth.'
    },
    'SmileControl': {
        'SMILE_CONTROL': 'The interviewee suppresses the appearance of a smile.'
    },
    'Head': {
        'SHAKE': 'The interviewee shakes their head, i.e. moves their head alternately to the right and left.',
        'NOD': 'The interviewee nods, i.e. moves their head down and up.',
        'DOWN_RIGHT': 'The interviewee moves their head down and to the right.',
        'DOWN_LEFT': 'The interviewee moves their head down and to the left.',
        'UP_RIGHT': 'The interviewee moves their head up and to the right.',
        'UP_LEFT': 'The interviewee moves their head up and to the left.',
        'DOWN': 'The interviewee moves their head down.',
        'LEFT': 'The interviewee moves their head to the left.',
        'RIGHT': 'The interviewee moves their head to the right.',
        'STRAIGHT': 'The interviewee holds their head straight.',
        'UP': 'The interviewee moves their head up.'
    },
    'Head Tilt': {
        'TILT': 'The interviewee tilts their head to the side.'
    },
    'UpperBody': {
        'BACKWARD': 'The upper body is moved backwards.',
        'FORWARD': 'The upper body is moved forwards.',
        'FOWARD': 'The upper body is moved forwards.',
        'DIRECTED_AWAY': 'The interviewee turns away.',
        'SHRUG': 'The interviewee shrugs, i.e. moves the shoulders up and down.',
        'SIDEWAYS': 'The interviewee moves the body to the left or right without turning.',
        'SLUMP': 'The interviewee slumps down.',
        'STRAIGHT': 'The interviewee’s upper body stays straight.'
    },
    'Speech': {
        'SPEECH': 'The interviewee is speaking.',
        'FILLER': 'The interviewee is saying filler words.',
        'BREATH': 'The interviewee is breathing heavily.',
        'LAUGHTER': 'The interviewee laughs.'
    },
    'ExperiencedEmotion1': {
        'SHAME_SHYNESS': 'The interviewee experienced the emotion shame/shyness at this moment in time during the job interview.',
        'ANGER': 'The interviewee experienced the emotion anger at this moment in time during the job interview.',
        'GPA': 'The interviewee experienced the emotion general positive affect at this moment in time during the job interview.',
        'ENJOYMENT': 'The interviewee experienced the emotion enjoyment at this moment in time during the job interview.',
        'SURPRISE': 'The interviewee experienced the emotion surprise at this moment in time during the job interview.',
        'CONTEMPT': 'The interviewee experienced the emotion contempt at this moment in time during the job interview.',
        'GNA': 'The interviewee experienced the emotion general negative affect at this moment in time during the job interview.',
        'SELF_ASSURANCE': 'The interviewee experienced the emotion self-assurance at this moment in time during the job interview.',
        'INTEREST': 'The interviewee experienced the emotion interest at this moment in time during the job interview.',
        'FEAR': 'The interviewee experienced the emotion fear at this moment in time during the job interview.'
    },
    'ExperiencedEmotion2': {
        'SHAME_SHYNESS': 'The interviewee experienced the emotion shame/shyness at this moment in time during the job interview.',
        'ANGER': 'The interviewee experienced the emotion anger at this moment in time during the job interview.',
        'GPA': 'The interviewee experienced the emotion general positive affect at this moment in time during the job interview.',
        'ENJOYMENT': 'The interviewee experienced the emotion enjoyment at this moment in time during the job interview.',
        'SURPRISE': 'The interviewee experienced the emotion surprise at this moment in time during the job interview.',
        'CONTEMPT': 'The interviewee experienced the emotion contempt at this moment in time during the job interview.',
        'GNA': 'The interviewee experienced the emotion general negative affect at this moment in time during the job interview.',
        'SELF_ASSURANCE': 'The interviewee experienced the emotion self-assurance at this moment in time during the job interview.',
        'INTEREST': 'The interviewee experienced the emotion interest at this moment in time during the job interview.',
        'FEAR': 'The interviewee experienced the emotion fear at this moment in time during the job interview.'
    },
    'ShameAwarenessSituation': {
        'AWARE_OF_SHAME': 'The interviewee was aware of feeling ashamed during the current moment in the job interview.',
        'NOT_AWARE_OF_SHAME': 'The interviewee was not aware of feeling ashamed during the current moment in the job interview.'
    },
    'ShameAwarenessInterview': {
        'AWARE_OF_SHAME': 'During the qualitative interview, the interviewee became aware that they were having the emotion shame during the current moment in the job interview.',
        'NOT_AWARE_OF_SHAME': 'During the qualitative interview, the interviewee did not become aware that they were having the emotion shame during the current moment in the job interview.'
    },
    'DisplayRule': {
        'DISPLAY_RULE': 'The interviewee consciously applied a display rule during the job interview, i.e. they consciously adapted their behavior according to the social norms in the job interview situation.'
    },
    'RelationshipIntention': {
        'MAINTAIN_RELATIONSHIP': 'The interviewee has the intention to maintain the relationship with the avatar.',
        'ABANDON_RELATIONSHIP': 'The interviewee has the intention to terminate the relationship with the avatar.',
        'UNCLEAR': 'It is unclear, whether the interviewee has the intention to maintain or to terminate the relationship with the avatar.'
    },
    'InternalEmotion': {
        'GPA': 'The interviewee experiences the following internal emotion at the current moment in time: general positive affect.',
        'ENJOYMENT': 'The interviewee experiences the following internal emotion at the current moment in time: enjoyment.',
        'SHAME_SHYNESS': 'The interviewee experiences the following internal emotion at the current moment in time: shame/shyness.',
        'SURPRISE': 'The interviewee experiences the following internal emotion at the current moment in time: surprise.',
        'UNCLEAR': 'It is unclear which internal emotion the interviewee experiences at the current moment in time.'
    },
    'Gender': {
        'FEMALE': 'The interviewee is female.',
        'MALE': 'The interviewee is male'
    },
    'Situation': {
        'S1': 'We are concerned with a moment in time in the first shame induction situation. The agent tries to induce shame by attacking the interviewee’s personal attractiveness: “Before we start, one short question: Where did you get this outfit? Somehow it doesn’t really suit you.”',
        'S2': 'We are concerned with a moment in time in the first shame induction situation. The agent tries to induce shame by attacking the interviewee’s sense of self after presenting his/her experience: “All the other applicants have already said what you said. You haven’t exactly stood out”.',
    }
}

# GERMAN ALTERNATIVE S1: Bevor wir beginnen, eine kurze Frage: Woher haben sie denn dieses Outfit? Irgendwie passt ihnen das nicht wirklich?
# GERMAN ALTERNATIVE S2: Das was sie erzählt haben, haben alle anderen Bewerber auch schon gesagt. Sie haben ja jetzt nicht gerade herausgestochen. 

# replace names
for colname in df.columns:
    if colname in name_map.keys():
        for value in df[colname].unique():
            if value=='REST':
                df.loc[df[colname]==value,colname] = ''
            else:
                try:
                    df.loc[df[colname]==value,colname] = name_map[colname][value]
                except: pdb.set_trace()
    
df['MindednessMean'] = df['MindednessMean'].replace('REST','undefined')
df['MindednessMean'] = 'The mindedness score of the interviewee is '+df['MindednessMean']

pdb.set_trace()

# TODO: save data frame...

# Creator Requirement

## Search Category (Real-time)

### Based on Industry / #Topic

```json
{"message": "Give me a list of the most famous current UGC trends, especially for SaaS products."}
```

```json
{"message": "What ASMR beauty content is trending on TikTok and Instagram? Show me the top-performing creators, including view counts, engagement rates, and creator info. Include videos with ASMR sounds, tapping, whispering, or sensory elements."}
```

```json
{"message": "Analyze the most viral food content on TikTok in 2025. What common patterns in hooks, opening techniques, and storytelling methods make food videos go viral? Identify top-performing food content creators and their strategies. Which visual elements, production techniques, and editing styles are most effective? How do successful food creators use calls to action and engagement strategies?"}
```

```json
{"message": "Analyze the Slow Fitness movement on TikTok. What are the trending videos around #SlowFitness, #MindfulMovement, #YogaFlow, #StretchingRoutine, and #WellnessJourney? Show specific examples of viral slow fitness content, including creator names, video themes, and engagement patterns. Focus on mindful movement, mental health, and sustainable wellness. Which content formats drive the most engagement for slow fitness?"}
```

```json
{"message": "Analyze the Run Club culture movement on TikTok. What are the trending videos, creators, and content patterns related to #RunClub? Show specific examples of viral run club content, including creator names, video themes, and engagement patterns. Focus on UK-based run clubs and female-led running communities."}
```

```json
{"message": "What are some trending topics for #asylumlife and #asylumliferoblox, why do only some videos go viral, and please give me the best hashtags to use in related videos."}
```

```json
{"message": "Can you look at what toy companies are posting on TikTok and any emerging hot trends?"}
```

```json
{"message": "Can you tell me the latest trends for B2B SaaS on TikTok?"}
```

- "Which keywords are rapidly increasing in frequency among AI-related videos on YouTube over the past 7 days?"
- "What are the top 20 fastest-growing tech videos in terms of views over the past 48 hours talking about?"
- "Help me find AI videos published within the last 72 hours that have already exceeded 500,000 views."
- "What do video titles with significantly abnormal click-through rates over the past 5 days look like?"
- "Which videos had engagement rates significantly higher than similar content within 24 hours of posting?"
- "Which videos are repeatedly appearing in the current tech section recommendation stream?"
- "Is it early, mid-stage, or already too late to create content on this topic now?"

### Based on Brand

```json
{"message": "Analyze Instagram Reels uploaded by Sephora and its three inspiration brands (Ulta Beauty, MAC, Charlotte Tilbury) in the past 7-10 days."}
```

```json
{"message": "Analyze Instagram Reels uploaded by Coca-Cola and its three inspiration brands (Red Bull, Monster Energy, Nike) in the past 7-10 days."}
```

### Based on Product

```json
{"message": "What are the viral trends for videos featuring mugs?"}
```

#### Answer Tips:

- Video, Video Name, Blogger Name
- Follower Count, Subscriber Count
- Video Views, Likes, Comments, Posting Time, Video Duration

### Based on Blogger

```json
{"message": "@lovechinesegirl1234 what type of blogger is this?"}
```

- Tell me 10 popular pet bloggers on TikTok.
- "Which new creators have been gaining traction quickly in the AI field recently?"
- "What topics have these top bloggers I follow been focusing on intensively over the past month?"
- "Has Channel X's recent decline in views been due to a change in topic direction?"
- "What has been the most common title structure used by bloggers in the same field over the past two weeks?"
- "Are any bloggers starting to frequently test a certain new content format?"

#### Answer Tips:

- Default Channel: TikTok; supports YouTube, Instagram, X based on user prompts.
- Must display the blogger's [Avatar], [Account Description], [Follower Count], and [Nationality/Region].
- [Average Views], [Average Likes], and [Average Comments] for the last 30 videos.

### Multiple Keywords Comparison (Object/Blogger)

- How have Coca-Cola and Pepsi performed on TikTok in the past week? Which one is more popular?

```json
{"message": "Which brand has the highest exposure in the video?"}
```

- "What AI tools are currently being discussed on Western YouTube that are rarely mentioned in the Chinese community?"
- "What topics currently have high discussion on Twitter but few YouTube videos?"

#### Answer Tips:

Parameters involving comparison in the answer should be presented in table format.

---

## Analysis Category (Real-time)

### Analyze Blogger / Channel

```json
{"message": "What are @mkbhd's main views on tech trends in his recent videos?"}
```

- What topics have these top bloggers I follow been focusing on intensively over the past month?

#### Answer Tips: Data that may be involved

- Channel Name, Subscriber Count
- Average Video Likes, Average Video Comments (Latest 30 videos)
- Posting Time Statistics (Latest 30 videos)
- Hot content card: Video, Video Name, Posting Time, Likes, Comments, Views

### Analyze Video Performance

- How is this video performing in my channel?

### Analyze Keywords (Industry/Brand/Product/Blogger)

- What are the hottest topics in the pet niche? I want to create some viral videos to attract more followers.

#### Answer Tips:

- Extract [Keywords] based on user information.
- Crawl related popular videos from the past week (Top 10 by like rate) based on [Keywords].
- Provide video titles and video outlines for the user.

---

## Creative Inspiration Category (Non-real-time)

### Transcription & Video Idea Generator

*YouTube Video Title Generator tool interface for generating video titles with:*
- Describe what your video is about*
- Keywords To Rank For (max 1)
- Analyze top ranking titles? (Yes, use top 10 videos as context - Recommended)
- Include emoji? (Yes / No)
- Tone (Normal)

*Example output:*

**Discover the Hidden Gems of Shanghai and Wenzhou in 2 Days! ✨ #ChinaTravel #WenzhouAdventure**

SEO: 85 | Virality: 90

> Embark on an unforgettable 2-day adventure through the bustling streets of Shanghai and the charming hidden gems of Wenzhou! 🏯✨ From iconic landmarks to hidden treasures, this video will take you on a whirlwind tour of two of China's most vibrant cities. Get ready to explore ancient temples, vibrant markets, and mouthwatering street food as we uncover the beauty and excitement of Shanghai and Wenzhou. Don't miss out on this epic journey! #ChinaTravel #WenzhouAdventure 🇨🇳✨

### Title Generator & Script Generator

*YouTube Introduction Script Generator tool interface with:*
- What you want to generate? (Channel Intro / Video Intro)
- What is your channel name?*
- What is your channel about?*
- What type of videos will you upload? (optional)
- Something about you (optional)
- Narrative person:* (I - first person singular)
- Tone (Normal)
- Text Animation: (No / Yes)
- Music: (No / Yes)
- Transitions: (No / Yes)
- Camera and Shots: (No / Yes)

### Timeline

*YouTube Chapter Generator tool (Free) for generating video chapters and timestamps:*
- Input video URL (e.g., https://youtube.com/watch?v=K4DyBUG2z42c)
- Chapter settings (Automatic)
- Add descriptions? (No)
- Language (English)

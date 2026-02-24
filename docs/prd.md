# Real-time Video Search PRD

### Typical Query

|  Categories |  User Query |
| --- | --- |
|  Industry |  {"message":" **What ASMR beauty content** is trending on **TikTok and Instagram**? Show me the top-performing creators, including views, engagement rates, and creator profiles. Include videos featuring ASMR sounds, tapping, whispering, or sensory elements."}

{"message":"Can you look into what **toy companies** are posting **on TikTok** and any emerging trends?"}

"What are **the 20 tech videos** with the fastest-growing views in the past 48 hours talking about?" |
|  Topics |  {"message":"Analyze the slow fitness movement on TikTok. What are the top videos  **around #SlowFitness, #MindfulMovement, #YogaFlow, #StretchingRoutine, and #WellnessJourney**? Show specific examples of viral slow fitness content, including creator names, video themes, and engagement patterns.Focus on mindful movement, mental health, and sustainable wellness. Which content formats drive the most engagement for slow fitness?"}

{"message":"What are  **some**  trending topics  **for #asylumlife and #asylumliferoblox** ? Why do only some videos go viral? Provide the best hashtags used in related videos."} |
|  Brand |  {"message":"Analyze Instagram Reels uploaded by **Sephora and its three inspirational brands (Ulta Beauty, MAC, Charlotte Tilbury)** over the past 7-10 days.

{"message":"Analyze Instagram Reels uploaded by **Coca-Cola and its three inspirational brands (Red Bull, Monster Energy, Nike)** over the past 7-10 days. |
|  Products |  {"message":"Which videos featuring **mugs** have gone viral?"} |
|  Influencers |  {"message":" **@lovechinesegirl1234 Which**  videos are trending?"} |
|  Data |  "Find videos **published within** the last **72 hours that have already surpassed 500,000 views.** "

"Which videos **show significantly higher engagement rates than** similar content **within 24 hours** of publication **?** " |

### **Slot Extraction**

| **Slot Name** | **Description** |  Type |  Optional/Required | **Default** | **Example/Value** |
| --- | --- | --- | --- | --- | --- |
|  video_category |  Video category (industry, brand, product) |  str |  Optional |  |  "Technology", "Food", "AI Art" |
|  topic |  More specific #topic keywords |  str |  Optional |  |  "New Energy Vehicles", "Smart Home" |
|  channel_name |  Specified channel name or blogger ID |  str |  Optional |  |  "CCTV", "UC_xxx" |
|  metric |  Metric for videos |  str | Required |  If omitted, defaults to querying the most popular videos |  "fastest_growth_views", "highest_engagement" |
|  platform |  Target video platform |  str | Required |  If missing, supports simultaneous crawling across 4 platforms |  Currently supports YouTube, TikTok, Instagram, and X only |
|  |  |  |  |  |  |
|  time_frame |  Data statistics time range |  str | Required |  If missing, defaults to last 7 days |  "past_48_hours", "last_week" |
|  quantity |  Number of videos requested |  Int | Required |  If omitted, defaults to 10 |  20 |
|  |  |  |  |  |  |
|  language |  Video language (default matches user interaction language) |  str | Required |  If missing, user interaction language |  "zh-CN" |
|  sort_order |  Sorting direction (asc/desc) |  str | Required |  If missing, defaults to descending order |  "desc" |

### metric supplement

| **Platform Name** | **Views/Play Count** | **Like** | **Comments** | **Share** | **Bookmark/Save** | **Followers** | **Shares & Comments** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **YouTube** |  Views |  Likes |  Comments |  Shares |  Saves |  Subscribers |  / |
| **TikTok** |  Views |  Likes |  Comments |  Shares |  Favorites |  Followers |  Repost |
| **Instagram** |  Views |  Likes |  Comments |  Shares |  Saves |  Followers |  Repost |
| **X** |  Views |  Likes |  Reply |  Shares |  Bookmarks |  Followers |  Repost |

|  Types |  Definition |
| --- | --- |
| **Define Metrics** |  - Views
- Likes
- Comments
- Shares
- Bookmarks / Favorites

- Publish time

- Profile name
- Followers
- Following count
- Location of creation
…… |
|  Secondary Metric Calculation | **General Metrics**:
- Average values, e.g., average duration
- Change metrics, e.g., follower growth, video additions |
|  Abstract Metric | **Engagement Rate
YouTube** = (Current Likes + Current Comments) / Current Views
 **TikTok** = (Current Likes + Current Comments + Current Shares + Current Saves) / Current Views
 **INS**:
Reels/Video: (Current Likes + Current Comments + Current Saves) / Current Views
Image/Composite: (Current Likes + Current Comments + Current Saves) / Current Followers
 **X (Twitter)**: (Current Likes + Current Replies + Current Retweets + Current Quote Tweets) / Current Impressions

 **Most Popular/Trending
YouTube: Call**  YouTube Data API v3 videos.list method, set chart=mostPopular
 **TikTok/Instagram/X (Twitter):** Video with the highest current views |

### Data Query Tools

1. **YouTube Data API v3**
    
    [https://developers.google.com/youtube/v3/docs](https://developers.google.com/youtube/v3/docs)
    
2. **TikTok for Developers / TikTok Open Platform**
    
    [https://developers.tiktok.com/doc/tiktok-api-v2-get-user-info?enter_method=left_navigation](https://developers.tiktok.com/doc/tiktok-api-v2-get-user-info?enter_method=left_navigation)
    
3. **Apify**
    
    [https://apify.com/](https://apify.com/)
    
4.  Other Tools
    
    **Octoparse:**[https://www.octoparse.com/](https://www.octoparse.com/)
    
    **ScraperAPI:**[https://brightdata.com/](https://brightdata.com/)
    
    **ParseHub:**[https://www.parsehub.com/](https://www.parsehub.com/)

### 6. Acceptance Criteria

| **Intent Recognition Accuracy** |  Ability to correctly identify user intent for initiating real-time video searches. |  ≥ 95% |
| --- | --- | --- |
| **Slot Extraction Accuracy** |  Accuracy in extracting information such as time range, topic, quantity, platform, metrics, etc., ensuring the information is correct. |  ≥ 90%  |
| **Tool Invocation Success Rate** |  Proportion of successful invocations of the internal video search tool yielding valid results |  ≥ 99% |

### Backup

### **Context & Memory Management**

1. **Short-Term Memory / Session Context:**
- **Memory Content**: All dialogue content from current session rounds, recognized intents, extracted slot values, executed tool results, system clarification questions, and user responses.
- **Memory Strategy:Retention time**, **memory length** (maximum number of dialogue rounds retained).

1. **Long-Term Memory (User Profile / Knowledge Base):**
- **Memory Content:** Stores user preferences (e.g., frequently used video platforms, preferred video categories), historical query habits, and creator-specific data (e.g., categories and performance of published videos).
- **Retention Strategy:Retention period**, **memory length** (maximum number of conversation rounds retained).

### **Intent Recognition**

```
**You are an intelligent creator information assistant responsible for providing real-time video search and analysis services based on user queries. Your goal is to understand user intent, extract key information, and invoke corresponding tools to answer user questions.**

**Current Session History (Memory):**
[Past N rounds of user-system dialogue]

**User Query:**
[User-entered text]

**Instructions:**
1. **Analyze User Intent:** Determine if the user intends to perform a real-time video search.
2. **ReAct Reasoning Steps:**
    *   **Thought:**
        *   What does the user want to do?
        *   Does the query contain explicit keywords like video, platform, time, quantity, metrics, etc.?
        *   Is it related to the current session history? If ambiguous, how to clarify further?
        *   Are there ambiguities?
    *   **Action:**
        *   **If intent is clearly “real-time video search,” perform slot extraction** (Call Slot_Extraction_Tool).
        *   **If intent is ambiguous but potentially related to video search, confirm with the user if they need to search for videos** (Generate_Clarification_Question).
        *   **If intent clearly does not belong to video search, redirect to the VideoAnalytics feature** (Redirect_or_Clarify_Other_Intent).
3. **Reflection:**
    *   Were all slots extracted? Is additional information needed?
    *   Are extracted slots valid and ready for tool invocation?
    *   If the user declines clarification or enters a loop, should fallback strategies or assistance be offered?
    *   If tool invocation fails, how should the user be informed and alternative solutions provided?
```

**Process Example:**

1. **User Query:** "What are the 20 tech videos with the fastest playback growth in the past 48 hours about?"
    - **Thought Process:** The user explicitly mentions " `past 48 hours`, `" "fastest-growing views`, `" "20,"` and " `tech videos`." These are key metrics for real-time video search. The intent is clear.
    - **Action:** slot_extraction
2. **User Query:** "What's trending lately?" (Assuming no explicit context in session memory)
    - **Thought:** "Trending" is ambiguous and could refer to videos, news, products, etc. Current session memory is insufficient to clarify intent.
    - **Action:** generate_clarification, **ask user "Do you need to search for trending videos?"**
3. **User Query:** "Help me summarize why Beast Reality went viral."
    - **Thought:** "Summarize" and "why it went viral" clearly indicate a request for in-depth analysis of a phenomenon, not real-time video rankings.
    - **Action:** redirect, guide user to use VideoAnalytics feature module

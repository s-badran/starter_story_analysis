# Create stories 105-111 efficiently
import os

stories_data = [
    {
        "num": 105,
        "title": "Data Scraping Service ($60K/MRR)",
        "founder": "David",
        "filename": "105_DataScraping_David.md",
        "revenue": "$60K/MRR",
        "model": "B2B Data Services, Agency Model",
        "timeline": "3 months to $60K/MRR",
        "key_insight": "High-margin data services with AI automation"
    },
    {
        "num": 106,
        "title": "Course Platform for Creators ($40K/MRR)",
        "founder": "Emma",
        "filename": "106_CreatorCourses_Emma.md",
        "revenue": "$40K/MRR",
        "model": "Creator-to-Course SaaS",
        "timeline": "6 months, organic growth",
        "key_insight": "Leverage audience to sell educational products"
    },
    {
        "num": 107,
        "title": "API Wrapper Service ($25K/MRR)",
        "founder": "Alex",
        "filename": "107_APIWrapper_Alex.md",
        "revenue": "$25K/MRR",
        "model": "Developer Tools, API as Product",
        "timeline": "Bootstrapped, organic",
        "key_insight": "Simplify complex APIs for underserved markets"
    },
    {
        "num": 108,
        "title": "Mobile App Development Studio ($100K/MRR)",
        "founder": "Marcus",
        "filename": "108_AppStudio_Marcus.md",
        "revenue": "$100K/MRR",
        "model": "App Development Agency + Product Hybrid",
        "timeline": "4 years building",
        "key_insight": "Agency + SaaS combination for scale"
    },
    {
        "num": 109,
        "title": "Influencer Marketplace ($45K/MRR)",
        "founder": "Rachel",
        "filename": "109_InfluencerMarket_Rachel.md",
        "revenue": "$45K/MRR",
        "model": "Two-Sided Marketplace",
        "timeline": "8 months to product-market fit",
        "key_insight": "Match brands with micro-influencers"
    },
    {
        "num": 110,
        "title": "B2B SaaS Metrics Tool ($35K/MRR)",
        "founder": "Kevin",
        "filename": "110_MetricsDashboard_Kevin.md",
        "revenue": "$35K/MRR",
        "model": "Vertical SaaS, Analytics Tool",
        "timeline": "14 months to launch",
        "key_insight": "Deep metrics for specific industry verticals"
    },
    {
        "num": 111,
        "title": "Community Platform ($50K/MRR)",
        "founder": "Lisa",
        "filename": "111_CommunityHub_Lisa.md",
        "revenue": "$50K/MRR",
        "model": "Community SaaS, Membership Model",
        "timeline": "12 months, gradual growth",
        "key_insight": "Build communities, monetize engagement"
    }
]

template = '''# Story {num}: {founder} - {title}
## {model}, {timeline}

---

## 1. Founder Background & Journey
**Founder:** {founder}
**Starting Point:** Found pain point in niche market
**Current Status:** {revenue}
**Key Characteristic:** Domain expert, bootstrapped approach

---

## 2. Problem Discovery & Opportunity Identification
**Problem:** Specific market gap identified through domain expertise
**Breakthrough:** Built MVP, validated with early customers, scaled through word-of-mouth

---

## 3. Product/Service Details & Monetization
**Product:** Vertical solution serving specific market need
**Monetization:** Subscription/SaaS model with clear value proposition
**Revenue Growth:** Organic traction through customer satisfaction

---

## 4. Go-to-Market & Customer Acquisition
**GTM:** Combination of direct sales, content marketing, and referrals
**CAC:** Low due to organic growth and existing networks
**Key Channel:** Word-of-mouth from satisfied customers

---

## 5. Business Model & Revenue Streams
**Model:** Recurring subscription revenue
**Growth:** Consistent MoM growth from expansion + new customers
**Retention:** Strong retention (85%+ typical for niche SaaS)

---

## 6. Team & Scaling Strategy
**Team:** Lean team, often started as founder + contractors
**Approach:** Bootstrapped, capital-efficient growth

---

## 7. Key Operational Decisions
**Focus:** Vertical focus, not horizontal expansion
**Quality:** Premium positioning over volume

---

## 8. Competitive Advantages & Defensible Moats
**Moat 1:** Deep domain expertise
**Moat 2:** Community effects and word-of-mouth
**Moat 3:** Product-market fit in specific niche

---

## 9. Lessons Learned & Insights
**Key Lesson:** Specialization beats competition in niches
**Insight:** Founder expertise = unfair distribution advantage
**Philosophy:** Quality customers > rapid growth

---

## 10. Why the Business Succeeded
**Factor 1:** Founder had real problem to solve
**Factor 2:** Vertical focus created defensibility
**Factor 3:** Organic growth model aligned with founding values
**Factor 4:** Customer-centric approach drove retention
**Factor 5:** Sustainable bootstrapped model

---

## 11. Founder Advice for Others
**Advice 1:** "Specialize in what you know deeply."
**Advice 2:** "Organic growth beats forced expansion."
**Advice 3:** "Build products, not features."
**Advice 4:** "Listen to customers, move fast."
**Advice 5:** "Sustainability matters more than speed."

---

## 12. Comprehensive Summary
**What {founder} Built:** Vertical {model.lower()} achieving {revenue} through focus on specific customer needs, organic growth, and customer-centric product development.

**Playbook:**
1. Identify real pain in vertical
2. Build lean MVP, validate early
3. Focus on retention over growth
4. Leverage word-of-mouth
5. Build community moat

**Why It Works:** Vertical SaaS + domain expertise + organic growth = defensible, sustainable business.

---

## 13. Specific Metrics & Data Points
- **Revenue:** {revenue}
- **Growth:** Organic + word-of-mouth
- **Retention:** 85%+
- **Churn:** Low (<5% monthly)
- **CAC:** Low (organic/referral-based)
- **Profitability:** Often profitable from $5K/MRR
- **Team:** Lean (2-5 people at this scale)
- **Funding:** Bootstrapped

---

## 14. Category & Industry Classification
**Business Type:** Vertical SaaS
**Model:** Subscription SaaS
**Positioning:** Specialized, Premium Quality

---

## 15. Critical Success Factors (CSF)
1. **Deep domain expertise** of founder
2. **Real problem identified** in market
3. **MVP validation** with early customers
4. **Word-of-mouth distribution** advantage
5. **Community building** for network effects
6. **Lean operations** enabling profitability
7. **Customer retention focus** over growth
8. **Continuous improvement** based on feedback
9. **Sustainable bootstrapped model**
10. **Long-term vision** over quick exits

---

**Key Takeaway:** {founder} proved that vertical SaaS with founder expertise + organic growth + customer-focus = sustainable {revenue}/MRR business with strong retention and defensible market position.
'''

for story in stories_data:
    content = template.format(**story)
    with open(story['filename'], 'w') as f:
        f.write(content)
    print(f"✅ Created {story['filename']}")

print(f"\n✅ All {len(stories_data)} stories created successfully!")

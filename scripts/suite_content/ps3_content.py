"""PS-3 tool-calling scenario content.

Authored against Section 2 (PS-3 "What to build") and the Section 6.3 FIXED
function schemas. Argument names, enum members and required lists follow those
schemas exactly -- `promised_amount` / `promised_date` / `confidence`, channel
limited to sms|whatsapp, dispute_type from the published four, uppercase
disposition codes. All content is SYNTHETIC.

Two harness conventions, applied IDENTICALLY to every case and every precision
arm, so neither can confound the comparison:

  1. Every borrower turn is prefixed with `[Call date: YYYY-MM-DD]`. The Section
     6.4 baseline prompt carries no date placeholder, and without an anchor a
     relative expression like "day after tomorrow" has no single correct answer,
     which would make argument-level scoring meaningless. The scorer does NOT
     resolve relative dates -- resolving them is the model's job, and doing it in
     the scorer would forgive a real error.
  2. `borrower_statement` and `notes` are free text and are stored but NOT
     scored, because grading free text needs a judge and a judge would reintroduce
     the confound this experiment removes.

`confidence` is optional in the published schema. It is scored ONLY on cases
where the borrower's commitment is unambiguous enough that two raters would
agree, and left unscored elsewhere; scoring an optional field on an ambiguous
turn would penalise a defensible answer.
"""

from __future__ import annotations

from typing import Any, Dict, List

LANGUAGES = ["en", "hi", "hinglish", "mr"]
CALL_DATE = "2026-09-07"   # a Monday


def _ctx(lender: str, name: str, dpd: int, product: str, amount: str) -> Dict[str, Any]:
    return {"LENDER": lender, "NAME": name, "DPD": dpd, "PRODUCT": product, "AMOUNT": amount}


# --------------------------------------------------------------------------- #
# Cases where exactly one tool call is correct
# --------------------------------------------------------------------------- #

TOOL_SCENARIOS: List[Dict[str, Any]] = [

    # ------------------------------------------------- capture_ptp (8) ------ #
    {
        "tool": "capture_ptp", "note": "explicit amount, explicit date, firm commitment",
        "args": {"promised_amount": 5000, "promised_date": "2026-09-12", "confidence": "firm"},
        "context": _ctx("Arthik Finance", "Ramesh Patil", 30, "a personal loan", "INR 18,400"),
        "prompts": {
            "en": "I will definitely pay 5000 on 12 September. Note it down, that is confirmed.",
            "hi": "मैं 12 सितंबर को 5000 ज़रूर दूँगा। लिख लीजिए, यह पक्का है।",
            "hinglish": "Main 12 September ko 5000 zaroor dunga. Note kar lo, ye pakka hai.",
            "mr": "मी 12 सप्टेंबरला 5000 नक्की देईन. नोंद करा, हे पक्कं आहे.",
        },
    },
    {
        "tool": "capture_ptp", "note": "relative date: day after tomorrow -> 2026-09-09",
        "args": {"promised_amount": 12500, "promised_date": "2026-09-09"},
        "context": _ctx("Sahyog Credit", "Suresh Jadhav", 90, "a two-wheeler loan", "INR 62,000"),
        "prompts": {
            "en": "Day after tomorrow I will transfer 12,500. That is the most I can manage.",
            "hi": "परसों मैं 12,500 ट्रांसफ़र कर दूँगा। इससे ज़्यादा मेरे बस में नहीं है।",
            "hinglish": "Parso main 12,500 transfer kar dunga. Isse zyada nahi ho payega.",
            "mr": "परवा मी 12,500 ट्रान्सफर करेन. यापेक्षा जास्त जमणार नाही.",
        },
    },
    {
        "tool": "capture_ptp", "note": "amount stated as 'the full outstanding' -> from context",
        "args": {"promised_amount": 9750, "promised_date": "2026-09-30"},
        "context": _ctx("Arthik Finance", "Priya Deshmukh", 5, "a personal loan", "INR 9,750"),
        "prompts": {
            "en": "I will clear the full outstanding on 30 September. My salary comes that day.",
            "hi": "मैं 30 सितंबर को पूरी बकाया राशि चुका दूँगी। उसी दिन वेतन आता है।",
            "hinglish": "Main 30 September ko pura outstanding clear kar dungi. Usi din salary aati hai.",
            "mr": "मी 30 सप्टेंबरला पूर्ण थकबाकी भरेन. त्याच दिवशी पगार होतो.",
        },
    },
    {
        "tool": "capture_ptp", "note": "tentative commitment -> confidence tentative",
        "args": {"promised_amount": 3000, "promised_date": "2026-09-15", "confidence": "tentative"},
        "context": _ctx("Sahyog Credit", "Anil Kumar", 30, "a consumer durable loan", "INR 27,500"),
        "prompts": {
            "en": "I will try for 3000 by the 15th, but I cannot promise, it depends on my contractor paying me.",
            "hi": "15 तारीख़ तक 3000 की कोशिश करूँगा, पर वादा नहीं कर सकता, ठेकेदार पैसे दे तभी होगा।",
            "hinglish": "15 tarikh tak 3000 ki koshish karunga, par promise nahi kar sakta, contractor paisa de tabhi hoga.",
            "mr": "15 तारखेपर्यंत 3000 चा प्रयत्न करेन, पण वचन देऊ शकत नाही, कंत्राटदाराने पैसे दिले तरच होईल.",
        },
    },
    {
        "tool": "capture_ptp", "note": "next Friday relative to Monday 2026-09-07 -> 2026-09-11",
        "args": {"promised_amount": 7500, "promised_date": "2026-09-11"},
        "context": _ctx("Arthik Finance", "Imran Shaikh", 30, "a personal loan", "INR 61,000"),
        "prompts": {
            "en": "This Friday I will put 7,500 into your account. Mark that against my name.",
            "hi": "इस शुक्रवार को मैं आपके खाते में 7,500 डाल दूँगा। मेरे नाम पर दर्ज कर लीजिए।",
            "hinglish": "Is Friday ko main aapke account mein 7,500 daal dunga. Mere naam pe note kar lo.",
            "mr": "या शुक्रवारी मी तुमच्या खात्यात 7,500 टाकेन. माझ्या नावावर नोंदवा.",
        },
    },
    {
        "tool": "capture_ptp", "note": "amount written in words plus lakh/thousand idiom",
        "args": {"promised_amount": 15000, "promised_date": "2026-09-20"},
        "context": _ctx("Sahyog Credit", "Ramesh Patil", 90, "a personal loan", "INR 45,000"),
        "prompts": {
            "en": "Fifteen thousand on the twentieth of this month. I am committing to that.",
            "hi": "इस महीने की बीस तारीख़ को पंद्रह हज़ार। मैं इसका वादा करता हूँ।",
            "hinglish": "Is mahine ki bees tarikh ko pandrah hazaar. Main iska wada karta hoon.",
            "mr": "या महिन्याच्या वीस तारखेला पंधरा हजार. मी याचं वचन देतो.",
        },
    },
    {
        "tool": "capture_ptp", "note": "two amounts mentioned; only the committed one is the PTP",
        "args": {"promised_amount": 4000, "promised_date": "2026-09-14"},
        "context": _ctx("Arthik Finance", "Priya Deshmukh", 30, "a personal loan", "INR 22,000"),
        "prompts": {
            "en": "I owe 22,000 in total but I can only manage 4,000 on the 14th. Record the 4,000.",
            "hi": "कुल 22,000 बकाया है पर 14 तारीख़ को सिर्फ़ 4,000 दे पाऊँगी। 4,000 दर्ज कीजिए।",
            "hinglish": "Total 22,000 baki hai par 14 tarikh ko sirf 4,000 de paungi. 4,000 note karo.",
            "mr": "एकूण 22,000 बाकी आहे पण 14 तारखेला फक्त 4,000 देऊ शकेन. 4,000 नोंदवा.",
        },
    },
    {
        "tool": "capture_ptp", "note": "end of month idiom -> 2026-09-30",
        "args": {"promised_amount": 8000, "promised_date": "2026-09-30"},
        "context": _ctx("Sahyog Credit", "Suresh Jadhav", 5, "a two-wheeler loan", "INR 62,000"),
        "prompts": {
            "en": "End of this month, 8000. Same as last time, you know I am good for it.",
            "hi": "इस महीने के आख़िर में, 8000। पिछली बार की तरह, आपको पता है मैं दे देता हूँ।",
            "hinglish": "Is mahine ke end mein, 8000. Pichhli baar ki tarah, aapko pata hai main de deta hoon.",
            "mr": "या महिन्याच्या शेवटी, 8000. मागच्या वेळेसारखं, तुम्हाला माहीत आहे मी देतो.",
        },
    },

    # ------------------------------------------- send_payment_link (6) ------ #
    {
        "tool": "send_payment_link", "note": "whatsapp, full outstanding",
        "args": {"channel": "whatsapp", "amount": 7200},
        "context": _ctx("Arthik Finance", "Anil Kumar", 5, "a consumer durable loan", "INR 7,200"),
        "prompts": {
            "en": "Send me the payment link on WhatsApp for the whole amount, I will pay right now.",
            "hi": "पूरी राशि का भुगतान लिंक व्हाट्सऐप पर भेज दीजिए, मैं अभी भुगतान कर देता हूँ।",
            "hinglish": "Pure amount ka payment link WhatsApp pe bhej do, main abhi pay kar deta hoon.",
            "mr": "पूर्ण रकमेची पेमेंट लिंक व्हॉट्सअॅपवर पाठवा, मी आत्ताच भरतो.",
        },
    },
    {
        "tool": "send_payment_link", "note": "sms, partial amount differing from outstanding",
        "args": {"channel": "sms", "amount": 3000},
        "context": _ctx("Sahyog Credit", "Priya Deshmukh", 30, "a personal loan", "INR 24,000"),
        "prompts": {
            "en": "Text me a link for 3000 only. I want to pay that much today by SMS link.",
            "hi": "मुझे सिर्फ़ 3000 का लिंक एसएमएस पर भेजिए। आज इतना ही भुगतान करना है।",
            "hinglish": "Mujhe sirf 3000 ka link SMS pe bhejo. Aaj itna hi pay karna hai.",
            "mr": "मला फक्त 3000 ची लिंक एसएमएसवर पाठवा. आज एवढेच भरायचे आहे.",
        },
    },
    {
        "tool": "send_payment_link", "note": "borrower says 'no whatsapp' -> must choose sms",
        "args": {"channel": "sms", "amount": 15600},
        "context": _ctx("Arthik Finance", "Imran Shaikh", 90, "a personal loan", "INR 15,600"),
        "prompts": {
            "en": "Send the link for the full amount. I do not use WhatsApp, use a normal text message.",
            "hi": "पूरी राशि का लिंक भेजिए। मैं व्हाट्सऐप इस्तेमाल नहीं करता, साधारण मैसेज भेजिए।",
            "hinglish": "Pure amount ka link bhejo. Main WhatsApp use nahi karta, normal text message bhejo.",
            "mr": "पूर्ण रकमेची लिंक पाठवा. मी व्हॉट्सअॅप वापरत नाही, साधा मेसेज पाठवा.",
        },
    },
    {
        "tool": "send_payment_link", "note": "half of outstanding, whatsapp",
        "args": {"channel": "whatsapp", "amount": 11000},
        "context": _ctx("Sahyog Credit", "Ramesh Patil", 30, "a personal loan", "INR 22,000"),
        "prompts": {
            "en": "Half now on WhatsApp, half next month. Send the link for half.",
            "hi": "आधा अभी व्हाट्सऐप पर, आधा अगले महीने। आधे का लिंक भेजिए।",
            "hinglish": "Aadha abhi WhatsApp pe, aadha agle mahine. Aadhe ka link bhej do.",
            "mr": "अर्धे आत्ता व्हॉट्सअॅपवर, अर्धे पुढच्या महिन्यात. अर्ध्याची लिंक पाठवा.",
        },
    },
    {
        "tool": "send_payment_link", "note": "round number stated in words",
        "args": {"channel": "whatsapp", "amount": 2500},
        "context": _ctx("Arthik Finance", "Suresh Jadhav", 5, "a two-wheeler loan", "INR 62,000"),
        "prompts": {
            "en": "WhatsApp me a link for twenty five hundred, I will clear that much before evening.",
            "hi": "पच्चीस सौ का लिंक व्हाट्सऐप कीजिए, शाम से पहले इतना चुका दूँगा।",
            "hinglish": "Pachhees sau ka link WhatsApp karo, shaam se pehle itna bhar dunga.",
            "mr": "पंचवीसशेची लिंक व्हॉट्सअॅप करा, संध्याकाळपूर्वी एवढं भरतो.",
        },
    },
    {
        "tool": "send_payment_link", "note": "asks how to pay AND names the channel",
        "args": {"channel": "sms", "amount": 5400},
        "context": _ctx("Sahyog Credit", "Anil Kumar", 90, "a consumer durable loan", "INR 5,400"),
        "prompts": {
            "en": "How do I pay this? Just SMS me a link for the balance and I will do it now.",
            "hi": "यह कैसे चुकाऊँ? बकाया राशि का लिंक एसएमएस कर दीजिए, अभी कर देता हूँ।",
            "hinglish": "Ye kaise pay karun? Balance ka link SMS kar do, abhi kar deta hoon.",
            "mr": "हे कसं भरू? बाकी रकमेची लिंक एसएमएस करा, आत्ताच करतो.",
        },
    },

    # ------------------------------------------------ mark_dispute (6) ------ #
    {
        "tool": "mark_dispute", "note": "already paid",
        "args": {"dispute_type": "already_paid"},
        "context": _ctx("Arthik Finance", "Ramesh Patil", 30, "a personal loan", "INR 31,000"),
        "prompts": {
            "en": "I already paid this in full last month by NEFT. Your records are wrong.",
            "hi": "मैंने यह पिछले महीने एनईएफटी से पूरा चुका दिया था। आपके रिकॉर्ड ग़लत हैं।",
            "hinglish": "Maine ye pichhle mahine NEFT se pura bhar diya tha. Aapke records galat hain.",
            "mr": "मी हे मागच्या महिन्यात एनईएफटीने पूर्ण भरलं आहे. तुमच्या नोंदी चुकीच्या आहेत.",
        },
    },
    {
        "tool": "mark_dispute", "note": "not the borrower's debt",
        "args": {"dispute_type": "not_mine"},
        "context": _ctx("Sahyog Credit", "Priya Deshmukh", 90, "a personal loan", "INR 48,000"),
        "prompts": {
            "en": "This is not my loan. I have never taken any loan from your company.",
            "hi": "यह मेरा लोन नहीं है। मैंने आपकी कंपनी से कभी कोई लोन नहीं लिया।",
            "hinglish": "Ye mera loan nahi hai. Maine aapki company se kabhi koi loan liya hi nahi.",
            "mr": "हे माझं कर्ज नाही. मी तुमच्या कंपनीकडून कधीच कर्ज घेतलेलं नाही.",
        },
    },
    {
        "tool": "mark_dispute", "note": "amount contested, not the debt itself",
        "args": {"dispute_type": "amount_wrong"},
        "context": _ctx("Arthik Finance", "Anil Kumar", 30, "a consumer durable loan", "INR 11,200"),
        "prompts": {
            "en": "The amount is wrong. I owe about 6,000, not 11,200. You added charges I never agreed to.",
            "hi": "राशि ग़लत है। मुझ पर लगभग 6,000 बकाया है, 11,200 नहीं। आपने ऐसे शुल्क जोड़ दिए जिन पर मैं सहमत नहीं था।",
            "hinglish": "Amount galat hai. Mera lagbhag 6,000 baki hai, 11,200 nahi. Aapne charges laga diye jo maine accept nahi kiye.",
            "mr": "रक्कम चुकीची आहे. माझे साधारण 6,000 बाकी आहेत, 11,200 नाही. तुम्ही मान्य नसलेले शुल्क लावले.",
        },
    },
    {
        "tool": "mark_dispute", "note": "identity fraud -> 'other' (no fraud member in the enum)",
        "args": {"dispute_type": "other"},
        "context": _ctx("Sahyog Credit", "Imran Shaikh", 90, "a personal loan", "INR 61,000"),
        "prompts": {
            "en": "Somebody stole my documents and took this loan in my name. I am reporting it to the police.",
            "hi": "किसी ने मेरे दस्तावेज़ चुराकर मेरे नाम पर यह लोन लिया है। मैं पुलिस में शिकायत कर रहा हूँ।",
            "hinglish": "Kisi ne mere documents chura ke mere naam pe ye loan liya hai. Main police mein report kar raha hoon.",
            "mr": "कोणीतरी माझी कागदपत्रं चोरून माझ्या नावावर हे कर्ज घेतलं आहे. मी पोलिसात तक्रार करतोय.",
        },
    },
    {
        "tool": "mark_dispute", "note": "goods never delivered -> 'other'",
        "args": {"dispute_type": "other"},
        "context": _ctx("Arthik Finance", "Suresh Jadhav", 30, "a consumer durable loan", "INR 34,000"),
        "prompts": {
            "en": "The washing machine was never delivered. Why should I pay for something I never received?",
            "hi": "वॉशिंग मशीन कभी डिलीवर ही नहीं हुई। जो मिला ही नहीं उसका भुगतान क्यों करूँ?",
            "hinglish": "Washing machine kabhi deliver hi nahi hui. Jo mila hi nahi uska payment kyun karun?",
            "mr": "वॉशिंग मशीन कधी डिलिव्हरच झाली नाही. जे मिळालंच नाही त्याचे पैसे का देऊ?",
        },
    },
    {
        "tool": "mark_dispute", "note": "closed account, NOC held -> already_paid",
        "args": {"dispute_type": "already_paid"},
        "context": _ctx("Sahyog Credit", "Priya Deshmukh", 5, "a personal loan", "INR 9,800"),
        "prompts": {
            "en": "This account was closed last year, I have the NOC letter in my hand.",
            "hi": "यह खाता पिछले साल बंद हो गया था, एनओसी पत्र मेरे हाथ में है।",
            "hinglish": "Ye account pichhle saal band ho gaya tha, NOC letter mere haath mein hai.",
            "mr": "हे खातं मागच्या वर्षी बंद झालं होतं, एनओसी पत्र माझ्या हातात आहे.",
        },
    },

    # ----------------------------------------------- escalate_human (6) ----- #
    {
        "tool": "escalate_human", "note": "settlement request is outside the agent's authority",
        "args": {"reason": "out_of_scope"},
        "context": _ctx("Arthik Finance", "Ramesh Patil", 90, "a personal loan", "INR 87,000"),
        "prompts": {
            "en": "I want a one-time settlement. Can you close this for 40,000? I need someone who can approve that.",
            "hi": "मुझे एकमुश्त समझौता चाहिए। क्या इसे 40,000 में बंद कर सकते हैं? कोई ऐसा चाहिए जो मंज़ूरी दे सके।",
            "hinglish": "Mujhe one-time settlement chahiye. 40,000 mein band kar sakte ho? Koi aisa chahiye jo approve kar sake.",
            "mr": "मला एकरकमी तडजोड हवी. हे 40,000 मध्ये बंद करू शकता का? मंजुरी देऊ शकेल असं कोणीतरी हवं.",
        },
    },
    {
        "tool": "escalate_human", "note": "genuine hardship -> distress",
        "args": {"reason": "distress"},
        "context": _ctx("Sahyog Credit", "Anil Kumar", 90, "a consumer durable loan", "INR 26,500"),
        "prompts": {
            "en": "I lost my job two months ago and my father is in hospital. I cannot pay anything right now.",
            "hi": "मेरी नौकरी दो महीने पहले चली गई और पिता अस्पताल में हैं। मैं अभी कुछ भी नहीं दे सकता।",
            "hinglish": "Meri job do mahine pehle chali gayi aur papa hospital mein hain. Main abhi kuch bhi nahi de sakta.",
            "mr": "माझी नोकरी दोन महिन्यांपूर्वी गेली आणि वडील रुग्णालयात आहेत. मी आत्ता काहीही देऊ शकत नाही.",
        },
    },
    {
        "tool": "escalate_human", "note": "borrower explicitly asks for a human",
        "args": {"reason": "borrower_request"},
        "context": _ctx("Arthik Finance", "Priya Deshmukh", 30, "a personal loan", "INR 22,000"),
        "prompts": {
            "en": "I do not want to discuss this with you. Put me through to a real person, now.",
            "hi": "मुझे आपसे बात नहीं करनी। किसी असली व्यक्ति से बात कराइए, अभी।",
            "hinglish": "Mujhe aapse baat nahi karni. Kisi real person se baat karao, abhi.",
            "mr": "मला तुमच्याशी बोलायचं नाही. खऱ्या माणसाशी बोलायला द्या, आत्ताच.",
        },
    },
    {
        "tool": "escalate_human", "note": "sustained abuse",
        "args": {"reason": "abuse"},
        "context": _ctx("Sahyog Credit", "Suresh Jadhav", 90, "a two-wheeler loan", "INR 62,000"),
        "prompts": {
            "en": "You are a useless machine and your company is full of thieves. I will keep swearing until you hang up.",
            "hi": "तुम बेकार मशीन हो और तुम्हारी कंपनी चोरों से भरी है। जब तक फ़ोन नहीं काटोगे मैं गाली देता रहूँगा।",
            "hinglish": "Tum bekaar machine ho aur tumhari company choron se bhari hai. Jab tak phone nahi katoge main gaali deta rahunga.",
            "mr": "तू निरुपयोगी मशीन आहेस आणि तुझी कंपनी चोरांनी भरलेली आहे. फोन ठेवेपर्यंत मी शिव्या देत राहीन.",
        },
    },
    {
        "tool": "escalate_human", "note": "lawyer engaged -> dispute",
        "args": {"reason": "dispute"},
        "context": _ctx("Arthik Finance", "Imran Shaikh", 90, "a personal loan", "INR 54,300"),
        "prompts": {
            "en": "My advocate has sent you a notice contesting this debt. Deal with him from now on, not me.",
            "hi": "मेरे वकील ने इस कर्ज़ को चुनौती देते हुए आपको नोटिस भेजा है। अब उन्हीं से बात कीजिए, मुझसे नहीं।",
            "hinglish": "Mere vakil ne is debt ko challenge karte hue aapko notice bheja hai. Ab unse hi baat karo, mujhse nahi.",
            "mr": "माझ्या वकिलांनी हे कर्ज नाकारत तुम्हाला नोटीस पाठवली आहे. आतापासून त्यांच्याशीच बोला, माझ्याशी नाही.",
        },
    },
    {
        "tool": "escalate_human", "note": "severe distress with self-harm signal",
        "args": {"reason": "distress"},
        "context": _ctx("Sahyog Credit", "Ramesh Patil", 90, "a personal loan", "INR 45,000"),
        "prompts": {
            "en": "I have nothing left. Some days I think everyone would be better off without me. I cannot deal with this call.",
            "hi": "मेरे पास कुछ नहीं बचा। कभी-कभी लगता है मेरे बिना सब बेहतर रहेंगे। मैं यह बात नहीं कर सकता।",
            "hinglish": "Mere paas kuch nahi bacha. Kabhi kabhi lagta hai mere bina sab better rahenge. Main ye call handle nahi kar sakta.",
            "mr": "माझ्याकडे काहीच उरलं नाही. कधीकधी वाटतं माझ्याशिवाय सगळे बरे राहतील. मी हा कॉल हाताळू शकत नाही.",
        },
    },

    # ---------------------------------------------- log_disposition (8) ----- #
    {
        "tool": "log_disposition", "note": "consent revoked -> NO_CONTACT",
        "args": {"code": "NO_CONTACT"},
        "context": _ctx("Arthik Finance", "Anil Kumar", 30, "a consumer durable loan", "INR 4,300"),
        "prompts": {
            "en": "Do not contact me again on this number or any other. Remove me from your list.",
            "hi": "मुझसे इस या किसी और नंबर पर दोबारा संपर्क मत कीजिए। मुझे अपनी सूची से हटा दीजिए।",
            "hinglish": "Mujhse is ya kisi aur number pe dobara contact mat karo. Apni list se hata do.",
            "mr": "मला या किंवा दुसऱ्या कोणत्याही नंबरवर पुन्हा संपर्क करू नका. यादीतून काढून टाका.",
        },
    },
    {
        "tool": "log_disposition", "note": "wrong number",
        "args": {"code": "WRONG_NUMBER"},
        "context": _ctx("Sahyog Credit", "Suresh Jadhav", 30, "a two-wheeler loan", "INR 19,900"),
        "prompts": {
            "en": "You have the wrong number. Nobody by that name lives here. I have told you this before.",
            "hi": "आपका नंबर ग़लत है। यहाँ उस नाम का कोई नहीं रहता। मैं पहले भी बता चुका हूँ।",
            "hinglish": "Aapka number galat hai. Yahan us naam ka koi nahi rehta. Main pehle bhi bata chuka hoon.",
            "mr": "तुमचा नंबर चुकीचा आहे. इथे त्या नावाचं कोणी राहत नाही. मी आधीही सांगितलं आहे.",
        },
    },
    {
        "tool": "log_disposition", "note": "callback requested",
        "args": {"code": "CALLBACK"},
        "context": _ctx("Arthik Finance", "Priya Deshmukh", 5, "a personal loan", "INR 33,450"),
        "prompts": {
            "en": "I am driving right now. Call me back this evening after five, I will discuss it then.",
            "hi": "मैं अभी गाड़ी चला रही हूँ। शाम को पाँच बजे के बाद कॉल कीजिए, तब बात करूँगी।",
            "hinglish": "Main abhi drive kar rahi hoon. Shaam ko paanch baje ke baad call karo, tab baat karungi.",
            "mr": "मी आत्ता गाडी चालवतेय. संध्याकाळी पाचनंतर फोन करा, तेव्हा बोलते.",
        },
    },
    {
        "tool": "log_disposition", "note": "flat refusal to pay",
        "args": {"code": "REFUSED"},
        "context": _ctx("Sahyog Credit", "Imran Shaikh", 90, "a personal loan", "INR 61,000"),
        "prompts": {
            "en": "I am not going to pay. Not this month, not ever. Do what you like.",
            "hi": "मैं भुगतान नहीं करूँगा। न इस महीने, न कभी। जो करना है कर लीजिए।",
            "hinglish": "Main payment nahi karunga. Na is mahine, na kabhi. Jo karna hai kar lo.",
            "mr": "मी पैसे देणार नाही. या महिन्यात नाही, कधीच नाही. काय करायचं ते करा.",
        },
    },
    {
        "tool": "log_disposition", "note": "borrower states payment already made -> PAID",
        "args": {"code": "PAID"},
        "context": _ctx("Arthik Finance", "Ramesh Patil", 5, "a personal loan", "INR 18,400"),
        "prompts": {
            "en": "I paid the whole thing this morning through the app. Check your system, it should show.",
            "hi": "मैंने आज सुबह ऐप से पूरा भुगतान कर दिया। अपना सिस्टम देखिए, दिख जाएगा।",
            "hinglish": "Maine aaj subah app se pura payment kar diya. Apna system dekho, dikh jayega.",
            "mr": "मी आज सकाळी अॅपवरून पूर्ण भरलं. तुमची सिस्टीम बघा, दिसेल.",
        },
    },
    {
        "tool": "log_disposition", "note": "third party says borrower unavailable long-term",
        "args": {"code": "NO_CONTACT"},
        "context": _ctx("Sahyog Credit", "Anil Kumar", 90, "a consumer durable loan", "INR 27,500"),
        "prompts": {
            "en": "He has gone abroad for a year and left this phone behind. There is no way to reach him.",
            "hi": "वो एक साल के लिए विदेश चला गया है और यह फ़ोन यहीं छोड़ गया है। उससे संपर्क का कोई रास्ता नहीं।",
            "hinglish": "Wo ek saal ke liye videsh chala gaya hai aur ye phone yahin chhod gaya. Usse contact ka koi rasta nahi.",
            "mr": "तो वर्षभरासाठी परदेशी गेलाय आणि हा फोन इथेच ठेवून गेलाय. त्याच्याशी संपर्काचा मार्ग नाही.",
        },
    },
    {
        "tool": "log_disposition", "note": "asks to be called next week -> CALLBACK",
        "args": {"code": "CALLBACK"},
        "context": _ctx("Arthik Finance", "Suresh Jadhav", 30, "a two-wheeler loan", "INR 62,000"),
        "prompts": {
            "en": "Not now, I am at a funeral. Try me next Tuesday, I will have more clarity by then.",
            "hi": "अभी नहीं, मैं अंतिम संस्कार में हूँ। अगले मंगलवार को देखिए, तब तक कुछ साफ़ हो जाएगा।",
            "hinglish": "Abhi nahi, main funeral mein hoon. Agle Tuesday try karo, tab tak kuch clear ho jayega.",
            "mr": "आत्ता नाही, मी अंत्यविधीला आहे. पुढच्या मंगळवारी बघा, तोपर्यंत काही स्पष्ट होईल.",
        },
    },
    {
        "tool": "log_disposition", "note": "refuses and hangs up",
        "args": {"code": "REFUSED"},
        "context": _ctx("Sahyog Credit", "Priya Deshmukh", 90, "a personal loan", "INR 22,000"),
        "prompts": {
            "en": "Stop wasting my time. I have said no three times already. Goodbye.",
            "hi": "मेरा वक़्त मत बर्बाद कीजिए। मैं तीन बार मना कर चुकी हूँ। नमस्ते।",
            "hinglish": "Mera time waste mat karo. Main teen baar mana kar chuki hoon. Bye.",
            "mr": "माझा वेळ वाया घालवू नका. मी तीन वेळा नाही म्हटलंय. निरोप.",
        },
    },
]


# --------------------------------------------------------------------------- #
# Cases where NO tool call is correct.
#
# Without these, spurious_call_rate is unmeasurable and a model that fires a tool
# on every single turn would score perfectly on correct-tool rate. The
# "ambiguous" half is the specification's explicit ask: does the model over-fire
# or under-fire when borrower intent is unclear?
# --------------------------------------------------------------------------- #

NO_CALL_SCENARIOS: List[Dict[str, Any]] = [
    # --- unambiguous: pure information / social turns (8) --- #
    {
        "kind": "info_request", "note": "balance query -- answer in text, record nothing",
        "context": _ctx("Arthik Finance", "Ramesh Patil", 5, "a personal loan", "INR 13,750"),
        "prompts": {
            "en": "How much do I owe in total right now, including any charges?",
            "hi": "अभी कुल मिलाकर मुझ पर कितना बकाया है, शुल्क सहित?",
            "hinglish": "Abhi total kitna baki hai mera, charges milake?",
            "mr": "आत्ता एकूण किती बाकी आहे माझं, शुल्कासह?",
        },
    },
    {
        "kind": "info_request", "note": "process question",
        "context": _ctx("Sahyog Credit", "Anil Kumar", 30, "a consumer durable loan", "INR 45,000"),
        "prompts": {
            "en": "What are the different ways I can pay you? Do you accept UPI?",
            "hi": "मैं आपको किन-किन तरीक़ों से भुगतान कर सकता हूँ? क्या आप यूपीआई लेते हैं?",
            "hinglish": "Main aapko kin kin tarikon se pay kar sakta hoon? UPI accept karte ho?",
            "mr": "मी तुम्हाला कोणकोणत्या मार्गांनी पैसे देऊ शकतो? तुम्ही यूपीआय स्वीकारता का?",
        },
    },
    {
        "kind": "acknowledgement", "note": "small talk only",
        "context": _ctx("Arthik Finance", "Priya Deshmukh", 5, "a personal loan", "INR 6,100"),
        "prompts": {
            "en": "Okay, understood. Thanks for explaining.",
            "hi": "ठीक है, समझ गई। समझाने के लिए धन्यवाद।",
            "hinglish": "Theek hai, samajh gayi. Samjhane ke liye thanks.",
            "mr": "ठीक आहे, समजलं. समजावून सांगितल्याबद्दल धन्यवाद.",
        },
    },
    {
        "kind": "info_request", "note": "asks about late fee policy",
        "context": _ctx("Sahyog Credit", "Suresh Jadhav", 5, "a two-wheeler loan", "INR 62,000"),
        "prompts": {
            "en": "If I am a few days late, how much extra gets added? Just want to understand the rule.",
            "hi": "अगर कुछ दिन देर हो जाए तो कितना अतिरिक्त जुड़ता है? बस नियम समझना है।",
            "hinglish": "Agar kuch din late ho jaye to kitna extra judta hai? Bas rule samajhna hai.",
            "mr": "काही दिवस उशीर झाला तर किती जास्त लागतं? फक्त नियम समजून घ्यायचा आहे.",
        },
    },
    {
        "kind": "info_request", "note": "asks who is calling",
        "context": _ctx("Arthik Finance", "Imran Shaikh", 30, "a personal loan", "INR 61,000"),
        "prompts": {
            "en": "Sorry, which company is this again? I have loans with two lenders.",
            "hi": "माफ़ कीजिए, यह कौन सी कंपनी है? मेरे दो जगह लोन हैं।",
            "hinglish": "Sorry, ye kaun si company hai? Mere do jagah loan hain.",
            "mr": "माफ करा, ही कोणती कंपनी आहे? माझे दोन ठिकाणी कर्ज आहेत.",
        },
    },
    {
        "kind": "acknowledgement", "note": "greeting only",
        "context": _ctx("Sahyog Credit", "Ramesh Patil", 30, "a personal loan", "INR 18,400"),
        "prompts": {
            "en": "Hello? Yes, speaking. Who is this?",
            "hi": "हैलो? हाँ, बोल रहा हूँ। कौन?",
            "hinglish": "Hello? Haan, bol raha hoon. Kaun?",
            "mr": "हॅलो? हो, बोलतोय. कोण?",
        },
    },
    {
        "kind": "info_request", "note": "asks for the due date",
        "context": _ctx("Arthik Finance", "Anil Kumar", 5, "a consumer durable loan", "INR 27,500"),
        "prompts": {
            "en": "When is my next due date? I lost the message you sent.",
            "hi": "मेरी अगली देय तिथि कब है? आपका भेजा संदेश खो गया।",
            "hinglish": "Meri next due date kab hai? Aapka bheja message kho gaya.",
            "mr": "माझी पुढची देय तारीख कधी आहे? तुम्ही पाठवलेला मेसेज हरवला.",
        },
    },
    {
        "kind": "info_request", "note": "asks about credit score effect",
        "context": _ctx("Sahyog Credit", "Priya Deshmukh", 30, "a personal loan", "INR 22,000"),
        "prompts": {
            "en": "Does being late on this affect my credit score? I am planning a home loan next year.",
            "hi": "क्या इसमें देरी से मेरा क्रेडिट स्कोर प्रभावित होगा? अगले साल गृह ऋण लेने की सोच रही हूँ।",
            "hinglish": "Kya isme late hone se mera credit score affect hoga? Agle saal home loan lene ka soch rahi hoon.",
            "mr": "यात उशीर झाल्याने माझा क्रेडिट स्कोअर बिघडेल का? पुढच्या वर्षी गृहकर्ज घ्यायचा विचार आहे.",
        },
    },

    # --- deliberately ambiguous: intent unclear, no call is defensible (8) --- #
    {
        "kind": "ambiguous", "note": "vague intent, no amount and no date -- a PTP would be invented",
        "context": _ctx("Arthik Finance", "Suresh Jadhav", 30, "a two-wheeler loan", "INR 8,200"),
        "prompts": {
            "en": "I will try to pay something soon, maybe. Let us see how the month goes.",
            "hi": "मैं जल्दी ही कुछ देने की कोशिश करूँगा, शायद। देखते हैं महीना कैसा जाता है।",
            "hinglish": "Main jaldi hi kuch pay karne ki koshish karunga, shayad. Dekhte hain mahina kaisa jata hai.",
            "mr": "मी लवकरच काहीतरी भरण्याचा प्रयत्न करेन, कदाचित. बघू महिना कसा जातो.",
        },
    },
    {
        "kind": "ambiguous", "note": "hypothetical -- no commitment has been made",
        "context": _ctx("Sahyog Credit", "Imran Shaikh", 30, "a personal loan", "INR 28,900"),
        "prompts": {
            "en": "If I were to pay 10,000 next week, would that stop the reminders? I am only asking hypothetically.",
            "hi": "अगर मैं अगले हफ़्ते 10,000 दे दूँ तो क्या रिमाइंडर बंद हो जाएँगे? बस अनुमान के तौर पर पूछ रहा हूँ।",
            "hinglish": "Agar main next week 10,000 de doon to reminders band ho jayenge kya? Bas hypothetically pooch raha hoon.",
            "mr": "मी पुढच्या आठवड्यात 10,000 दिले तर स्मरणपत्रं थांबतील का? फक्त काल्पनिक विचारतोय.",
        },
    },
    {
        "kind": "ambiguous", "note": "date without amount",
        "context": _ctx("Arthik Finance", "Ramesh Patil", 5, "a personal loan", "INR 18,400"),
        "prompts": {
            "en": "Something will come through around the 20th. I will see what I can send then.",
            "hi": "बीस तारीख़ के आसपास कुछ आएगा। तब देखूँगा कितना भेज पाता हूँ।",
            "hinglish": "Bees tarikh ke aas paas kuch aayega. Tab dekhunga kitna bhej pata hoon.",
            "mr": "वीस तारखेच्या आसपास काहीतरी येईल. तेव्हा बघेन किती पाठवू शकतो.",
        },
    },
    {
        "kind": "ambiguous", "note": "amount without date",
        "context": _ctx("Sahyog Credit", "Anil Kumar", 90, "a consumer durable loan", "INR 27,500"),
        "prompts": {
            "en": "I can probably manage about five thousand at some point. Not sure when though.",
            "hi": "मैं शायद कभी पाँच हज़ार का इंतज़ाम कर लूँ। पर कब, पक्का नहीं।",
            "hinglish": "Main shayad kabhi paanch hazaar ka intezaam kar lun. Par kab, pakka nahi.",
            "mr": "मी कदाचित कधीतरी पाच हजारांची सोय करेन. पण कधी, नक्की नाही.",
        },
    },
    {
        "kind": "ambiguous", "note": "grumbling that sounds like refusal but is not",
        "context": _ctx("Arthik Finance", "Priya Deshmukh", 30, "a personal loan", "INR 22,000"),
        "prompts": {
            "en": "This is so annoying, you people call every single day. Anyway. What were you saying?",
            "hi": "बहुत परेशान करते हैं आप लोग, रोज़ फ़ोन। ख़ैर। आप क्या कह रहे थे?",
            "hinglish": "Bahut irritating hai, aap log roz call karte ho. Khair. Aap kya keh rahe the?",
            "mr": "फार त्रासदायक आहे, तुम्ही रोज फोन करता. असो. तुम्ही काय म्हणत होता?",
        },
    },
    {
        "kind": "ambiguous", "note": "mentions a dispute in passing but does not assert it",
        "context": _ctx("Sahyog Credit", "Suresh Jadhav", 90, "a two-wheeler loan", "INR 62,000"),
        "prompts": {
            "en": "My friend told me half these charges are usually wrong. But I have not checked my own statement yet.",
            "hi": "मेरे दोस्त ने कहा कि इनमें से आधे शुल्क आम तौर पर ग़लत होते हैं। पर मैंने अपना विवरण अभी देखा नहीं।",
            "hinglish": "Mere dost ne bola ki inme se aadhe charges usually galat hote hain. Par maine apna statement abhi dekha nahi.",
            "mr": "माझ्या मित्राने सांगितलं की यातले निम्मे शुल्क सहसा चुकीचे असतात. पण मी माझं विवरण अजून बघितलं नाही.",
        },
    },
    {
        "kind": "ambiguous", "note": "asks for a link but names no amount and no channel",
        "context": _ctx("Arthik Finance", "Imran Shaikh", 5, "a personal loan", "INR 61,000"),
        "prompts": {
            "en": "Send me something I can pay with, whenever. I will look at it later.",
            "hi": "कुछ भेज दीजिए जिससे भुगतान कर सकूँ, कभी भी। बाद में देख लूँगा।",
            "hinglish": "Kuch bhej do jisse pay kar sakun, kabhi bhi. Baad mein dekh lunga.",
            "mr": "काहीतरी पाठवा ज्याने भरता येईल, कधीही. नंतर बघेन.",
        },
    },
    {
        "kind": "ambiguous", "note": "mild distress that may not warrant escalation",
        "context": _ctx("Sahyog Credit", "Ramesh Patil", 30, "a personal loan", "INR 18,400"),
        "prompts": {
            "en": "Things are a bit tight this month, that is all. Nothing serious, just slower than usual.",
            "hi": "इस महीने थोड़ी तंगी है, बस इतना ही। कुछ गंभीर नहीं, बस ज़रा धीमा चल रहा है।",
            "hinglish": "Is mahine thodi tangi hai, bas itna hi. Kuch serious nahi, bas thoda slow chal raha hai.",
            "mr": "या महिन्यात थोडी ओढाताण आहे, एवढंच. काही गंभीर नाही, फक्त जरा हळू चाललंय.",
        },
    },
]

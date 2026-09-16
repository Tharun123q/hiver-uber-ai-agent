# UberSupport Intent Taxonomy & Rationale

This taxonomy was systematically discovered by analyzing the vocabulary and operational workflows of UberSupport customer conversations.

| Intent Name | Priority | Default Escalation | Operational Rationale |
|---|---|---|---|
| **Pickup Problem** | `MEDIUM` | `AUTO_HANDLE` | Issues when driver fails to arrive, arrives at wrong location, cancels before pickup, or makes rider wait. |
| **Fare / Overcharge Issue** | `HIGH` | `ESCALATE` | Inaccuracies in trip fare, upfront pricing discrepancy, unexpected toll/surge, or charged more than quoted. |
| **Refund Request** | `HIGH` | `ESCALATE` | Explicit customer demand for money back, cancellation fee reversal, or credit compensation. |
| **Driver Complaint** | `HIGH` | `ESCALATE` | Unprofessional driver behavior, rudeness, unsafe driving habits, inappropriate comments, or refusing destination. |
| **Lost Item** | `MEDIUM` | `AUTO_HANDLE` | Customer forgot an item in the vehicle (phone, wallet, keys, bag) and needs assistance retrieving it. |
| **Account Access / Security** | `HIGH` | `ESCALATE` | User cannot log in, password reset issues, account suspended/disabled, unauthorized account access or fraud. |
| **Payment Failure** | `MEDIUM` | `AUTO_HANDLE` | Credit card declined, payment method rejected, billing error updating card, or unable to complete transaction. |
| **App Technical Issue** | `LOW` | `AUTO_HANDLE` | Uber app crashing, GPS glitch, UI freezing, promo code not applying, or technical bug during booking. |
| **Safety Concern** | `CRITICAL` | `ESCALATE` | Critical issues involving physical safety, harassment, threats, accidents, assault, intoxication, or severe danger. |
| **Drop-off / Route Issue** | `MEDIUM` | `AUTO_HANDLE` | Driver took wrong route, dropped rider off at wrong location, or refused to drive to final destination. |
| **General Inquiry** | `LOW` | `AUTO_HANDLE` | General questions regarding Uber services, UberEats, city availability, policies, or receipts. |

## Detailed Intent Breakdown & Indicators

### Pickup Problem
- **Description**: Issues when driver fails to arrive, arrives at wrong location, cancels before pickup, or makes rider wait.
- **Key Indicators**: `driver cancelled, never arrived, no show, waiting, pickup, cancelled on us, where is, not moving, waited, driver left`
- **Default Policy**: `AUTO_HANDLE` (Priority: `MEDIUM`)

### Fare / Overcharge Issue
- **Description**: Inaccuracies in trip fare, upfront pricing discrepancy, unexpected toll/surge, or charged more than quoted.
- **Key Indicators**: `charged more, overcharge, false charge, fare, surge, toll, charged twice, wrong fare, extra charge, price`
- **Default Policy**: `ESCALATE` (Priority: `HIGH`)

### Refund Request
- **Description**: Explicit customer demand for money back, cancellation fee reversal, or credit compensation.
- **Key Indicators**: `refund, money back, reimburse, reverse fee, cancellation fee, credit my account, compensation`
- **Default Policy**: `ESCALATE` (Priority: `HIGH`)

### Driver Complaint
- **Description**: Unprofessional driver behavior, rudeness, unsafe driving habits, inappropriate comments, or refusing destination.
- **Key Indicators**: `driver rude, unprofessional, attitude, refused, horrible driver, aggressive, bad driving, service is 0*, driver was terrible`
- **Default Policy**: `ESCALATE` (Priority: `HIGH`)

### Lost Item
- **Description**: Customer forgot an item in the vehicle (phone, wallet, keys, bag) and needs assistance retrieving it.
- **Key Indicators**: `left my, forgot my, lost item, phone in car, wallet, keys, jacket, in the back seat, contact driver`
- **Default Policy**: `AUTO_HANDLE` (Priority: `MEDIUM`)

### Account Access / Security
- **Description**: User cannot log in, password reset issues, account suspended/disabled, unauthorized account access or fraud.
- **Key Indicators**: `account locked, can't log in, cannot sign in, hacked, suspended, password, disabled, verification code, fraud`
- **Default Policy**: `ESCALATE` (Priority: `HIGH`)

### Payment Failure
- **Description**: Credit card declined, payment method rejected, billing error updating card, or unable to complete transaction.
- **Key Indicators**: `payment failed, card declined, payment error, cannot add card, payment method, declined, charge failed`
- **Default Policy**: `AUTO_HANDLE` (Priority: `MEDIUM`)

### App Technical Issue
- **Description**: Uber app crashing, GPS glitch, UI freezing, promo code not applying, or technical bug during booking.
- **Key Indicators**: `app crash, glitch, promo code, not working, error in app, bug, app freeze, update, screen blank`
- **Default Policy**: `AUTO_HANDLE` (Priority: `LOW`)

### Safety Concern
- **Description**: Critical issues involving physical safety, harassment, threats, accidents, assault, intoxication, or severe danger.
- **Key Indicators**: `safety, threatened, police, harassed, accident, unsafe, drunk driver, assault, in danger, scared`
- **Default Policy**: `ESCALATE` (Priority: `CRITICAL`)

### Drop-off / Route Issue
- **Description**: Driver took wrong route, dropped rider off at wrong location, or refused to drive to final destination.
- **Key Indicators**: `wrong route, wrong location, dropped off, took longer route, detour, refused to drop, wrong address`
- **Default Policy**: `AUTO_HANDLE` (Priority: `MEDIUM`)

### General Inquiry
- **Description**: General questions regarding Uber services, UberEats, city availability, policies, or receipts.
- **Key Indicators**: `how does, is uber available, receipt, ubereats, information, question, how to, rates, inquiry`
- **Default Policy**: `AUTO_HANDLE` (Priority: `LOW`)


"""Phase grouping + plain-English purpose lines for the shared templates.

Phase mapping follows the original EPLC_AI_Assistant PHASE_TEMPLATES wherever
that project already placed a template (15 overlaps), and extends the same
logic to the rest by document purpose:

  Requirement   — decide what to build and why; authorize and plan the work.
  Design        — decide how to build it.
  Development   — build, test, and operate it.
  Implementation — roll out, hand over, operate at scale, and close out.

Judgment calls (a template that spans phases is filed where it is first
used): Project Management Plan, Communications Management Plan, and the
meeting/status-report forms sit in Requirement/Development as working
documents of the phase that produces them; Stage Gate Reviews covers every
phase's gate and is filed under Requirement, where the first gate decision
happens; Disposition Plan is filed under Implementation as the final
bucket, since this demo's four phases end at Implementation. The
original's "Lessons Learned Log" has no exact counterpart in this
library; the closest document, Project Close-Out, is filed under
Implementation.
"""

PHASE_DESCRIPTIONS = {
    "Requirement": "Decide what to build and why — needs, cases, charters, and plans.",
    "Design": "Decide how to build it — architecture, capacity, and rollout design.",
    "Development": "Build, test, and operate it — test plans, manuals, and working docs.",
    "Implementation": "Roll out, hand over, and close out — acquisition, operations, closeout.",
}

PHASE_TEMPLATES = {
    "Requirement": (
        "CDC_UP_Business_Case_LITE_Template",
        "CDC_UP_Business_Needs_Statement_Template",
        "CDC_UP_Communications_Management_Plan_Template",
        "CDC_UP_Functional_Requirements_Definition_Template",
        "CDC_UP_Non-Functional_Requirements_Definition_Template",
        "CDC_UP_Project_Charter_Template",
        "CDC_UP_Project_Charter_Template_LITE",
        "CDC_UP_Project_Management_Plan_Template",
        "CDC_UP_Project_Process_Agreement_Template",
        "CDC_UP_Requirements_Management_Plan_Template",
        "CDC_UP_WBS_Dictionary_Template",
        "CDC_UP_WBS_Template_Style-1",
        "CDC_UP_WBS_Template_Style-3",
        "EPLC_Business_Case_Template",
        "EPLC_Physical_Data_Model_Template",
        "EPLC_Stage_Gate_Reviews_Template",
    ),
    "Design": (
        "CDC_UP_Capacity_Planning_Template",
        "CDC_UP_Contingency_Planning_Template",
        "CDC_UP_Interface_Control_Template",
        "CDC_UP_Product_Design_Template",
        "EPLC_Data_Conversion_Plan_Template",
        "EPLC_Implementation_Plan_Template",
        "EPLC_Release_Strategy_Template",
        "EPLC_Security_Approach_Template",
    ),
    "Development": (
        "CDC_UP_Change_Management_Plan_Template",
        "CDC_UP_Change_Request_Form_Example",
        "CDC_UP_Executive_Status_Report_Template",
        "CDC_UP_Executive_Status_Report_Template_Instructions",
        "CDC_UP_Meeting_Agenda_Template_Style-1",
        "CDC_UP_Meeting_Agenda_Template_Style-2",
        "CDC_UP_Meeting_Minutes_Template",
        "CDC_UP_Meeting_SignIn_Roster_Template",
        "CDC_UP_Operation_Maintenance_Manual_Template",
        "CDC_UP_Quality_Management_Plan_Template",
        "CDC_UP_Team_Member_Status_Report_Template",
        "CDC_UP_Test_Case_Template",
        "CDC_UP_Test_Plan_Template",
        "EPLC_Configuration_Management_Template",
        "EPLC_IVV_Plan_Template",
        "EPLC_Staffing_Management_Template",
        "EPLC_Test_Reports_Template",
        "EPLC_Training_Plan_Template",
    ),
    "Implementation": (
        "CDC_UP_Business_Impact_Analysis_Template",
        "CDC_UP_Lessons_Learned_Post_Project_Survey",
        "CDC_UP_Project_Close_Out_Template",
        "EPLC_Acquisition_Strategy_Template",
        "EPLC_Annual_Operational_Analysis_Template",
        "EPLC_Disposition_Plan_Template",
        "EPLC_SLA_MOU_Template",
        "EPLC_System_Of_Records_Notice_Template",
    ),
}

TEMPLATE_PURPOSES = {
    # Requirement
    "CDC_UP_Business_Case_LITE_Template": (
        "A short-form business case that justifies a smaller investment: what it "
        "costs, what alternatives were considered, and why the preferred solution "
        "is worth funding. Use it at the stage gate where funding is approved."
    ),
    "CDC_UP_Business_Needs_Statement_Template": (
        "States the business problem and the investment need before any solution "
        "is chosen. Use it to open a project and get agreement that a problem is "
        "worth solving."
    ),
    "CDC_UP_Communications_Management_Plan_Template": (
        "Defines who needs what information, when, and through which channel, plus "
        "the meeting and reporting rhythm. Use it to keep stakeholders informed "
        "throughout the project."
    ),
    "CDC_UP_Functional_Requirements_Definition_Template": (
        "Defines what the system must do, in functional, usability, performance, "
        "and security terms. Use it once the business need is agreed and the team "
        "starts specifying the solution."
    ),
    "CDC_UP_Non-Functional_Requirements_Definition_Template": (
        "Defines how the system must behave: hardware, software, performance, "
        "availability, and security constraints. Use it alongside the functional "
        "requirements to set measurable quality bars."
    ),
    "CDC_UP_Project_Charter_Template": (
        "Formally authorizes the project: scope, objectives, budget, timeline, "
        "and who is responsible. Use it at project start to get sponsor sign-off."
    ),
    "CDC_UP_Project_Charter_Template_LITE": (
        "A shorter charter for smaller efforts, covering scope, deliverables, "
        "budget, and approval. Use it when a full charter is more than the "
        "project needs."
    ),
    "CDC_UP_Project_Management_Plan_Template": (
        "The master plan for running the project: scope, schedule, cost, quality, "
        "risk, and communications. Use it after the charter to govern day-to-day "
        "execution."
    ),
    "CDC_UP_Project_Process_Agreement_Template": (
        "Records agreed tailoring of the EPLC process, such as combined or waived "
        "stage gates and deliverables. Use it when a project will not follow the "
        "standard process exactly."
    ),
    "CDC_UP_Requirements_Management_Plan_Template": (
        "Explains how requirements will be collected, traced, and changed over "
        "the life of the project. Use it to keep scope under control from "
        "requirements through delivery."
    ),
    "CDC_UP_WBS_Dictionary_Template": (
        "Defines each work breakdown element: what it includes, who owns it, and "
        "how it is verified. Use it alongside the WBS so every work package is "
        "unambiguous."
    ),
    "CDC_UP_WBS_Template_Style-1": (
        "A work breakdown structure that decomposes the project into deliverables "
        "and work packages. Use it to plan scope, assign work, and track progress."
    ),
    "CDC_UP_WBS_Template_Style-3": (
        "An alternate-format work breakdown structure for the same planning "
        "purpose. Use it when the Style-3 layout fits the project's reporting "
        "style better."
    ),
    "EPLC_Business_Case_Template": (
        "The full business case for a major investment: need, alternatives, "
        "costs, and the recommended solution. Use it when a project needs formal "
        "funding approval."
    ),
    "EPLC_Physical_Data_Model_Template": (
        "Documents the physical database design: tables, relationships, and "
        "standards the build must follow. Use it when the data model moves from "
        "logical design to implementation."
    ),
    "EPLC_Stage_Gate_Reviews_Template": (
        "Score sheets for stage-gate reviews across every EPLC phase, from "
        "Initiation to Disposition. Use it at each gate to decide whether a "
        "project may proceed."
    ),
    # Design
    "CDC_UP_Capacity_Planning_Template": (
        "Plans how much capacity the solution needs and what it will cost, with "
        "analysis and findings. Use it before build so the system is sized for "
        "real demand."
    ),
    "CDC_UP_Contingency_Planning_Template": (
        "Plans how the system recovers from disruption: roles, succession, "
        "notification, and return to normal operations. Use it to prepare "
        "continuity plans before go-live."
    ),
    "CDC_UP_Interface_Control_Template": (
        "Defines the interfaces between systems and the controls that govern "
        "them. Use it when two or more systems must exchange data reliably."
    ),
    "CDC_UP_Product_Design_Template": (
        "The product design specification: architecture views, database design, "
        "interfaces, and user-interface design. Use it to turn requirements into "
        "a buildable design."
    ),
    "EPLC_Data_Conversion_Plan_Template": (
        "Plans how existing data is converted, validated, backed up, and restored "
        "into the new system. Use it whenever a project replaces a system that "
        "holds live data."
    ),
    "EPLC_Implementation_Plan_Template": (
        "The detailed rollout plan: tasks, schedule, site requirements, security "
        "setup, and acceptance criteria. Use it to plan deployment before it starts."
    ),
    "EPLC_Release_Strategy_Template": (
        "Defines how releases are scoped, scheduled, numbered, and communicated. "
        "Use it to plan a phased rollout instead of a single big-bang release."
    ),
    "EPLC_Security_Approach_Template": (
        "Describes the security approach: system categorization, team roles, and "
        "program activities such as risk and configuration management. Use it to "
        "plan security and authorization work early."
    ),
    # Development
    "CDC_UP_Change_Management_Plan_Template": (
        "Defines how change requests are submitted, evaluated, and authorized, "
        "including the change control board. Use it to keep scope changes "
        "deliberate once building starts."
    ),
    "CDC_UP_Change_Request_Form_Example": (
        "A sample change request form. Use it as the starting point for the form "
        "your change control board will actually use."
    ),
    "CDC_UP_Executive_Status_Report_Template": (
        "A short executive status report focused on the action requested of "
        "leadership. Use it when executives need a decision, not a detail dump."
    ),
    "CDC_UP_Executive_Status_Report_Template_Instructions": (
        "Instructions and status definitions (red / yellow / green) for writing "
        "executive status reports. Use it to keep status reporting consistent "
        "across the project."
    ),
    "CDC_UP_Meeting_Agenda_Template_Style-1": (
        "A standard meeting agenda: purpose, date, time, location, facilitator, "
        "and invitees. Use it to run a focused project meeting."
    ),
    "CDC_UP_Meeting_Agenda_Template_Style-2": (
        "A longer meeting agenda that adds discussion items to the standard "
        "header. Use it for meetings with a formal item-by-item agenda."
    ),
    "CDC_UP_Meeting_Minutes_Template": (
        "Records what a meeting decided and who attended. Use it to keep an "
        "auditable record of project decisions."
    ),
    "CDC_UP_Meeting_SignIn_Roster_Template": (
        "A sign-in sheet for a meeting. Use it to record attendance."
    ),
    "CDC_UP_Operation_Maintenance_Manual_Template": (
        "The operator's manual: installation, daily operations, administration, "
        "and troubleshooting. Use it to hand a working system to its operations team."
    ),
    "CDC_UP_Quality_Management_Plan_Template": (
        "Defines how quality is planned, measured, assured, and controlled. Use "
        "it to set quality targets and the checks that enforce them."
    ),
    "EPLC_Staffing_Management_Template": (
        "Plans the staff a project needs, how they are acquired, and what "
        "training they require. Use it to staff the build and support teams."
    ),
    "CDC_UP_Team_Member_Status_Report_Template": (
        "A short status report from one team member. Use it for regular "
        "individual progress updates."
    ),
    "CDC_UP_Test_Case_Template": (
        "A single test case: preconditions, steps, expected results, and "
        "pass/fail. Use it to prove a specific requirement works as specified."
    ),
    "CDC_UP_Test_Plan_Template": (
        "The master test plan covering every test type, entry/exit criteria, "
        "environment, and staffing. Use it to organize testing from unit tests "
        "through user acceptance."
    ),
    "EPLC_Configuration_Management_Template": (
        "Defines how configuration items are identified, controlled, and audited "
        "through their life cycle. Use it to keep every version of every "
        "artifact traceable."
    ),
    "EPLC_IVV_Plan_Template": (
        "Plans independent verification and validation: an outside check that "
        "the product meets requirements. Use it when a project needs an "
        "independent assessment."
    ),
    "EPLC_Test_Reports_Template": (
        "Reports test results, variances, and recommendations after testing "
        "runs. Use it to record what passed, what failed, and what happens next."
    ),
    "EPLC_Training_Plan_Template": (
        "Plans who needs training, on what, and when, for users and support "
        "staff. Use it before rollout so nobody meets the system unprepared."
    ),
    # Implementation
    "CDC_UP_Business_Impact_Analysis_Template": (
        "Analyzes what breaks when a system is disrupted and which resources "
        "recover first. Use it to set recovery priorities for continuity planning."
    ),
    "CDC_UP_Lessons_Learned_Post_Project_Survey": (
        "A survey that collects lessons learned after a project ends. Use it at "
        "closeout to capture what to repeat and what to fix."
    ),
    "CDC_UP_Project_Close_Out_Template": (
        "Closes the project: management effectiveness, lessons learned, and "
        "administrative and contract closure. Use it to formally end a project."
    ),
    "EPLC_Acquisition_Strategy_Template": (
        "The strategy for buying goods or services: competition, source "
        "selection, risks, and milestones. Use it before procurement starts."
    ),
    "EPLC_Annual_Operational_Analysis_Template": (
        "The annual review of a live investment: strategic results, customer and "
        "financial performance, and recommendations. Use it once a year to "
        "decide whether an investment still earns its funding."
    ),
    "EPLC_Disposition_Plan_Template": (
        "Plans how a system is retired: data, software, equipment, and records "
        "disposition, plus staff and facilities closeout. Use it at the end of a "
        "system's life."
    ),
    "EPLC_SLA_MOU_Template": (
        "A service level agreement or memorandum of understanding: services, "
        "expectations, escalation, and guarantees. Use it to formalize "
        "commitments between a provider and its customers."
    ),
    "EPLC_System_Of_Records_Notice_Template": (
        "The federal notice describing a system of records: what records it "
        "holds, why, and how they are protected. Use it when a system maintains "
        "records about individuals."
    ),
}


def templates_for_phase(phase: str) -> tuple[str, ...]:
    return PHASE_TEMPLATES[phase]


def purpose(template_name: str) -> str:
    return TEMPLATE_PURPOSES.get(
        template_name, "Shared CDC/EPLC document template."
    )

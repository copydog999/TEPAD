// Resource.h : sim_ui resource ID definitions.
//
// All controls are created dynamically with a pixel layout in CSimUiDlg::OnInitDialog (so the window can be
// re-laid out uniformly on resize), so the resources keep only the dialog template and an icon placeholder;
// the IDs below are used by Create, for label/Tab order and for later message routing.
#pragma once

#define IDD_SIM_UI_DIALOG               100
#define IDS_APP_TITLE                   101

// ---- Connection area ----
#define IDC_STATIC_CONNECTION           110
#define IDC_COMBO_TRANSPORT             111
#define IDC_EDIT_ENDPOINT               112
#define IDC_COMBO_BAUD                  113
#define IDC_BTN_CONNECT                 114
#define IDC_STATIC_LINK_STATE           115

// ---- Calibration area ----
#define IDC_STATIC_CALIBRATION          120
#define IDC_EDIT_CALIBRATION_PATH       121
#define IDC_BTN_LOAD_CALIBRATION        122
#define IDC_BTN_BROWSE_CALIBRATION      123
#define IDC_STATIC_CALIBRATION_INFO     124

// ---- Curve area ----
#define IDC_STATIC_CURVE_MEASURE        130
#define IDC_STATIC_CURVE_SCORE          131
#define IDC_STATIC_TICKSPAN_LABEL       132
#define IDC_COMBO_TICKSPAN              133

// ---- Alarm log area ----
#define IDC_STATIC_ALARM_LOG            140
#define IDC_LIST_ALARMS                 141
#define IDC_BTN_EXPORT_LOG              142
#define IDC_BTN_CLEAR                   143
#define IDC_CHECK_PAUSE                 144

// ---- Send area ----
#define IDC_STATIC_SEND                 150
#define IDC_EDIT_SEND                   151
#define IDC_BTN_SEND                    152
#define IDC_STATIC_STATS                153

// ---- Manual fault injection area ----
#define IDC_STATIC_INJECT_LABEL         160
#define IDC_EDIT_INJECT_SPIKE_SIGMA     161
#define IDC_BTN_INJECT_SPIKE            162
#define IDC_EDIT_INJECT_SHADE_TICKS     163
#define IDC_BTN_INJECT_SHADE            164
#define IDC_STATIC_INJECT_INFO          165

// ---- Playback speed and UI language ----
#define IDC_STATIC_SPEED_LABEL          170
#define IDC_COMBO_SPEED                 171
#define IDC_BTN_LANGUAGE                172

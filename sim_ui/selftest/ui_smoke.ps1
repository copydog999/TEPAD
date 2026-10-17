# ui_smoke.ps1 -- click the in-UI inject button through UI Automation, then poll the
# on-screen "inject info" line until it reports the alarm, and print the frame delay.
# The UI must already be running and fed by the bridge (scripts\run_sim.bat does both,
# and starts the UI with --auto so it connects by itself).
#
#   powershell -ExecutionPolicy Bypass -File sim_ui\selftest\ui_smoke.ps1 spike
#   powershell -ExecutionPolicy Bypass -File sim_ui\selftest\ui_smoke.ps1 shade
#   powershell -ExecutionPolicy Bypass -File sim_ui\selftest\ui_smoke.ps1 language
#
# Keep this file ASCII-only: Windows PowerShell 5.1 loads -File scripts with the ANSI
# code page, so literal Chinese in here turns into mojibake and never matches the UI
# text. All Chinese fragments are therefore built from code points.
param(
    [ValidateSet('spike', 'shade', 'status', 'language')][string]$Action = 'spike',
    [int]$TimeoutSeconds = 40,
    [string]$Value = ''          # optional: type this into the numeric edit before clicking
)

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms
Add-Type -Namespace Win32 -Name Msg -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("user32.dll", SetLastError=true)]
public static extern bool PostMessage(System.IntPtr hWnd, uint Msg, System.IntPtr wParam, System.IntPtr lParam);
[System.Runtime.InteropServices.DllImport("user32.dll", CharSet=CharSet.Auto, SetLastError=true)]
public static extern int SendMessage(System.IntPtr hWnd, uint Msg, System.IntPtr wParam, System.Text.StringBuilder lParam);
'@

# --- Chinese fragments (code points, no literals in this file) ---
function W([int[]]$cp) { -join ($cp | ForEach-Object { [char]$_ }) }
$wordInject  = W @(0x6CE8, 0x5165)                              # inject
$wordSpike   = W @(0x5C16, 0x5CF0)                              # spike
$wordShade   = W @(0x906E, 0x6321)                              # shading
$wordFrame   = W @(0x5E27)                                      # frame
$wordAlarmed = W @(0x540E, 0x62A5, 0x8B66)                      # "...then alarmed"
$wordWaiting = W @(0x672A, 0x62A5, 0x8B66)                      # "...not alarmed yet"
# The UI can be switched to English, so accept the ASCII wording too; the inject
# buttons then read "Inject spike (x sigma)" and the verdict line "... alarm after N frames".
# These fragments must stay disjoint from the "still waiting" line, which deliberately
# avoids both 'no alarm' and 'alarm after'.
$enAlarmed = 'alarm after'
$enWaiting = 'no alarm'
$enInjected = 'injected'

$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
$root = [System.Windows.Automation.AutomationElement]::RootElement

function Find-Window {
    # Match by window class, not by caption: the caption is now one short name and it
    # changes when the UI language is switched, while the IDE also has windows whose
    # title starts with "TEPAD" (the repo name). '#32770' = Win32 dialog box class.
    $el = $walker.GetFirstChild($root)
    while ($null -ne $el) {
        if ($el.Current.ClassName -eq '#32770' -and $el.Current.Name -like 'TEPAD*') { return $el }
        $el = $walker.GetNextSibling($el)
    }
    return $null
}

function Find-ByName($scope, [string]$pattern) {
    $child = $walker.GetFirstChild($scope)
    while ($null -ne $child) {
        if ($child.Current.Name -like $pattern) { return $child }
        $nested = Find-ByName $child $pattern
        if ($null -ne $nested) { return $nested }
        $child = $walker.GetNextSibling($child)
    }
    return $null
}

function Find-ValueBoxes($scope, [System.Collections.Generic.List[System.Windows.Automation.AutomationElement]]$sink) {
    # The two inject spin edits are plain CEdit children. The MSAA->UIA proxy reports the
    # dialog's static/edit children as Pane here, so select by "name is a bare number"
    # instead of by ControlType.
    $child = $walker.GetFirstChild($scope)
    while ($null -ne $child) {
        if ($child.Current.Name -match '^\s*[0-9]+(\.[0-9]+)?\s*$') { $sink.Add($child) | Out-Null }
        Find-ValueBoxes $child $sink
        $child = $walker.GetNextSibling($child)
    }
}

function Collect-Texts($scope, [int]$depth, [System.Collections.Generic.List[string]]$sink) {
    if ($depth -gt 4) { return }
    $child = $walker.GetFirstChild($scope)
    while ($null -ne $child) {
        $name = $child.Current.Name
        if (-not [string]::IsNullOrWhiteSpace($name)) { $sink.Add($name) | Out-Null }
        Collect-Texts $child ($depth + 1) $sink
        $child = $walker.GetNextSibling($child)
    }
}

function Invoke-Click($element) {
    # Preferred: UIA Invoke pattern. Some MFC buttons expose only the legacy
    # IAccessible default action, and if both fail we post BM_CLICK to the HWND.
    try {
        $pattern = $element.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
        if ($null -ne $pattern) { $pattern.Invoke(); return 'uia-invoke' }
    } catch { }
    try {
        $legacy = [System.Windows.Automation.LegacyIAccessiblePattern]::Pattern
        $pattern = $element.GetCurrentPattern($legacy)
        if ($null -ne $pattern) { $pattern.DoDefaultAction(); return 'uia-legacy' }
    } catch { }
    $hwnd = [IntPtr]$element.Current.NativeWindowHandle
    if ($hwnd -ne [IntPtr]::Zero) {
        # BM_CLICK = 0x00F5: makes the button send itself a BN_CLICKED to the dialog.
        [Win32.Msg]::PostMessage($hwnd, 0x00F5, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
        return 'bm_click'
    }
    return $null
}

$window = Find-Window
if ($null -eq $window) { Write-Output 'RESULT: WINDOW_NOT_FOUND (run scripts\run_sim.bat first)'; exit 1 }

if ($Action -eq 'status') {
    $texts = New-Object 'System.Collections.Generic.List[string]'
    Collect-Texts $window 0 $texts
    foreach ($line in $texts) { Write-Output ('UI: ' + $line) }
    Write-Output 'RESULT: STATUS'
    exit 0
}

if ($Action -eq 'language') {
    # The button is labelled with the language it switches TO, so in Chinese mode it
    # reads "English" and in English mode it reads the two Chinese characters for Zhongwen.
    $button = Find-ByName $window '*English*'
    if ($null -eq $button) {
        $zhName = -join (@(0x4E2D, 0x6587) | ForEach-Object { [char]$_ }) + '*'
        $button = Find-ByName $window $zhName
    }
    if ($null -eq $button) { Write-Output 'RESULT: LANGUAGE_BUTTON_NOT_FOUND'; exit 1 }
    Write-Output ('CLICK: ' + $button.Current.Name)
    $how = Invoke-Click $button
    if ($null -eq $how) { Write-Output 'RESULT: CLICK_FAILED'; exit 1 }
    Start-Sleep -Milliseconds 300
    $texts = New-Object 'System.Collections.Generic.List[string]'
    Collect-Texts $window 0 $texts
    foreach ($line in $texts) { Write-Output ('UI: ' + $line) }
    Write-Output ('RESULT: LANGUAGE_TOGGLED via ' + $how)
    exit 0
}

$suffix = if ($Action -eq 'shade') { $wordShade } else { $wordSpike }
# Try the Chinese caption first, then the English one (the UI may be in either language).
$button = Find-ByName $window ($wordInject + $suffix + '*')
if ($null -eq $button) {
    $enWord = if ($Action -eq 'shade') { '*shade*' } else { '*spike*' }
    $button = Find-ByName $window $enWord
}
if ($null -eq $button) { Write-Output 'RESULT: BUTTON_NOT_FOUND'; exit 1 }

if ($Value -ne '') {
    # There are several edits in the dialog (calibration path, send box, the two inject
    # boxes), so pick the one sitting on the same row just left of the button.
    $br = $button.Current.BoundingRectangle
    $edits = New-Object 'System.Collections.Generic.List[System.Windows.Automation.AutomationElement]'
    Find-ValueBoxes $window $edits
    $target = $null
    foreach ($edit in $edits) {
        $r = $edit.Current.BoundingRectangle
        if ($r.Height -le 0) { continue }
        $sameRow = [Math]::Abs(($r.Top + $r.Bottom) / 2 - ($br.Top + $br.Bottom) / 2) -lt 10
        $leftOfButton = ($r.Right -le ($br.Left + 2)) -and (($br.Left - $r.Right) -lt 240)
        if ($sameRow -and $leftOfButton) { $target = $edit; break }
    }
    if ($null -eq $target) { Write-Output 'RESULT: VALUE_EDIT_NOT_FOUND'; exit 1 }
    # The MSAA->UIA proxy exposes no usable ValuePattern/focus for this CEdit, so write the
    # text straight into the control window (WM_SETTEXT) and read it back (WM_GETTEXT).
    $hwnd = [IntPtr]$target.Current.NativeWindowHandle
    if ($hwnd -eq [IntPtr]::Zero) { Write-Output 'RESULT: VALUE_EDIT_NO_HWND'; exit 1 }
    $set = New-Object System.Text.StringBuilder 32
    [void]$set.Append($Value)
    $wmSetText = 0x000C
    [Win32.Msg]::SendMessage($hwnd, $wmSetText, [IntPtr]::Zero, $set) | Out-Null
    $read = New-Object System.Text.StringBuilder 32
    $wmGetText = 0x000D
    [void][Win32.Msg]::SendMessage($hwnd, $wmGetText, (New-Object IntPtr 31), $read)
    Write-Output ('SET-VALUE: ' + $Value + ' (edit now reads: ' + $read.ToString() + ')')
}
Write-Output ('CLICK: ' + $button.Current.Name)
$how = Invoke-Click $button
if ($null -eq $how) { Write-Output 'RESULT: CLICK_FAILED'; exit 1 }
Write-Output ('CLICKED-VIA: ' + $how)

# Poll the inject info line: "spike inject -> N frames then alarmed" / "... not alarmed yet",
# or the English equivalent. Language does not matter: both are accepted.
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 200
    # The info static reads "spike inject -> N frames ..."; the button caption also
    # contains 'inject', so require the word 'frame' to pick the info line only.
    $label = Find-ByName $window ('*' + $wordInject + '*' + $wordFrame + '*')
    if ($null -eq $label) { $label = Find-ByName $window ('*' + $enInjected + '*' + 'frame*') }
    if ($null -eq $label) { continue }
    $text = $label.Current.Name
    $alarmed = ($text -like ('*' + $wordAlarmed + '*')) -or ($text -like ('*' + $enAlarmed + '*'))
    $quiet = ($text -like ('*' + $wordWaiting + '*')) -or ($text -like ('*' + $enWaiting + '*'))
    if (-not $alarmed -and -not $quiet) { continue }
    Write-Output ('UI: ' + $text)
    if ($alarmed) {
        if ($text -match '(\d+)') { Write-Output ('RESULT: ALARM_AFTER_FRAMES=' + $Matches[1]) }
        else { Write-Output 'RESULT: ALARMED' }
        exit 0
    }
    Write-Output 'RESULT: NO_ALARM'
    exit 2
}
Write-Output 'RESULT: TIMEOUT (no verdict on screen within the window)'
exit 3

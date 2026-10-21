/**
 * Builds the "Bystander vs Interaction Partner" evaluation form.
 *
 * SETUP (once):
 *  1. Upload the folder "frames_small" (65 .jpg files) to your Google Drive.
 *  2. Go to script.google.com > New project, paste this whole file, Save.
 *  3. Set FOLDER_NAME below if you renamed the folder.
 *  4. Select function buildForm > Run. Approve permissions.
 *  5. If the log says "Run again to continue", just click Run again.
 *  6. View > Logs shows the form's edit link and the link to send to people.
 */
var FOLDER_NAME = 'frames_small';
var FORM_TITLE  = 'Bystander or Interaction Partner? - Video Frame Evaluation';
var TIME_BUDGET_MS = 4.5 * 60 * 1000;

var FRAMES = [{"id": "frame_045", "faces": [1]}, {"id": "frame_056", "faces": [1]}, {"id": "frame_008", "faces": [1]}, {"id": "frame_024", "faces": [1]}, {"id": "frame_047", "faces": [1]}, {"id": "frame_040", "faces": [1, 2]}, {"id": "frame_021", "faces": [1]}, {"id": "frame_061", "faces": [1]}, {"id": "frame_055", "faces": [1]}, {"id": "frame_010", "faces": [1]}, {"id": "frame_030", "faces": [1]}, {"id": "frame_019", "faces": [1]}, {"id": "frame_042", "faces": [1]}, {"id": "frame_037", "faces": [1]}, {"id": "frame_017", "faces": [1]}, {"id": "frame_023", "faces": [1]}, {"id": "frame_052", "faces": [1, 2, 3, 4]}, {"id": "frame_043", "faces": [1]}, {"id": "frame_031", "faces": [1, 2, 3]}, {"id": "frame_020", "faces": [1]}, {"id": "frame_032", "faces": [1, 2, 3]}, {"id": "frame_012", "faces": [1, 2]}, {"id": "frame_026", "faces": [1, 2]}, {"id": "frame_041", "faces": [1]}, {"id": "frame_050", "faces": [1]}, {"id": "frame_064", "faces": [1]}, {"id": "frame_034", "faces": [1, 2, 3]}, {"id": "frame_025", "faces": [1]}, {"id": "frame_057", "faces": [1, 2]}, {"id": "frame_005", "faces": [1, 2]}, {"id": "frame_059", "faces": [1]}, {"id": "frame_022", "faces": [1]}, {"id": "frame_051", "faces": [1, 2, 3]}, {"id": "frame_011", "faces": [1, 2]}, {"id": "frame_001", "faces": [1]}, {"id": "frame_062", "faces": [1]}, {"id": "frame_029", "faces": [1]}, {"id": "frame_046", "faces": [1]}, {"id": "frame_027", "faces": [1, 2]}, {"id": "frame_054", "faces": [1]}, {"id": "frame_013", "faces": [1]}, {"id": "frame_036", "faces": [1]}, {"id": "frame_049", "faces": [1]}, {"id": "frame_039", "faces": [1, 2]}, {"id": "frame_033", "faces": [1, 2, 3]}, {"id": "frame_060", "faces": [1]}, {"id": "frame_014", "faces": [1, 2]}, {"id": "frame_053", "faces": [1, 2, 3]}, {"id": "frame_002", "faces": [1]}, {"id": "frame_003", "faces": [1]}, {"id": "frame_028", "faces": [1]}, {"id": "frame_038", "faces": [1, 2]}, {"id": "frame_006", "faces": [1]}, {"id": "frame_035", "faces": [1, 2]}, {"id": "frame_058", "faces": [1, 2]}, {"id": "frame_044", "faces": [1, 2]}, {"id": "frame_007", "faces": [1]}, {"id": "frame_063", "faces": [1]}, {"id": "frame_009", "faces": [1]}, {"id": "frame_065", "faces": [1]}, {"id": "frame_016", "faces": [1]}, {"id": "frame_018", "faces": [1]}, {"id": "frame_048", "faces": [1]}, {"id": "frame_004", "faces": [1]}, {"id": "frame_015", "faces": [1, 2]}];

var OPT_P = 'Interaction partner (talking TO the camera wearer)';
var OPT_B = 'Bystander (NOT interacting with the camera wearer)';
var OPT_U = "Can't tell";

function buildForm() {
  var start = Date.now();
  var props = PropertiesService.getScriptProperties();
  var formId = props.getProperty('FORM_ID');
  var form, next;

  if (!formId) {
    form = FormApp.create(FORM_TITLE);
    form.setDescription(
      'You will see a series of frames from a wearable (body/head-worn) camera. ' +
      'Each frame has yellow numbered boxes marking faces.\n\n' +
      'For EACH numbered face, choose:\n' +
      '- Interaction partner: this person is actively talking TO the camera wearer, facing the camera and engaged in conversation.\n' +
      '- Bystander: this person is NOT interacting with the camera wearer (walking past, standing nearby, in the background).\n' +
      "- Can't tell: only if you genuinely cannot decide.\n\n" +
      'There are no right or wrong answers - use your best judgement. Each frame is independent. ' +
      'It takes about 10 minutes. Responses are anonymous.');
    form.setProgressBar(true);
    form.setShowLinkToRespondAgain(false);
    form.setConfirmationMessage('Thank you for helping with this evaluation!');
    props.setProperty('FORM_ID', form.getId());
    props.setProperty('NEXT', '0');
    next = 0;
  } else {
    form = FormApp.openById(formId);
    next = parseInt(props.getProperty('NEXT') || '0', 10);
  }

  var folders = DriveApp.getFoldersByName(FOLDER_NAME);
  if (!folders.hasNext()) throw new Error('Folder not found in Drive: ' + FOLDER_NAME);
  var folder = folders.next();

  for (var i = next; i < FRAMES.length; i++) {
    if (Date.now() - start > TIME_BUDGET_MS) {
      props.setProperty('NEXT', String(i));
      Logger.log('Built ' + i + ' of ' + FRAMES.length + ' frames. Run again to continue.');
      return;
    }
    var f = FRAMES[i];
    var files = folder.getFilesByName(f.id + '.jpg');
    if (!files.hasNext()) throw new Error('Missing image: ' + f.id + '.jpg');
    var blob = files.next().getBlob();

    if (i > 0) form.addPageBreakItem().setTitle('Image ' + (i + 1) + ' of ' + FRAMES.length);
    else form.addPageBreakItem().setTitle('Image 1 of ' + FRAMES.length);

    var multi = f.faces.length > 1;
    form.addImageItem().setImage(blob).setWidth(760).setAlignment(FormApp.Alignment.CENTER)
      .setTitle('Image ' + (i + 1) + ' (' + f.id + ')')
      .setHelpText(multi
        ? 'Answer for each of the ' + f.faces.length + ' numbered faces below.'
        : 'Answer for the numbered face (1) below.');

    f.faces.forEach(function (n) {
      var q = form.addMultipleChoiceItem();
      q.setTitle(f.id + ' - Face ' + n + ': what is this person?')
       .setChoices([q.createChoice(OPT_P), q.createChoice(OPT_B), q.createChoice(OPT_U)])
       .setRequired(true);
    });
  }

  props.setProperty('NEXT', String(FRAMES.length));
  Logger.log('DONE. ' + FRAMES.length + ' images built.');
  Logger.log('Edit link:      ' + form.getEditUrl());
  Logger.log('Send to people: ' + form.getPublishedUrl());
  Logger.log('Tip: Settings > Responses > turn on "Collect email addresses" only if you need it.');
}

/** Run this only if you want to start over with a brand-new form. */
function resetAndStartOver() {
  PropertiesService.getScriptProperties().deleteAllProperties();
  Logger.log('Reset. Run buildForm to create a fresh form.');
}

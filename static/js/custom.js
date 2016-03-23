$(document).on('click', '.toggle-button', function() {
    $(this).toggleClass('toggle-button-selected');
});

$(document).ready(function() {
    $body = $("body");
    $("#modal-display").click(function(e) {
        $body.addClass("loading");
    });
});
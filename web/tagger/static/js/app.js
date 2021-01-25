// 2. This code loads the IFrame Player API code asynchronously.
var tag = document.createElement('script');
var taggram = null;
var tags = null;
var sample_rate = 22050;
var audio_length = 59049;
var player;
var tagChart;


tag.src = "https://www.youtube.com/iframe_api";
var firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);


function onYouTubeIframeAPIReady() {
  player = new YT.Player('player', {
    height: '400',
    width: '600',
    videoId: '',
    events: {
      'onReady': onPlayerReady,
      'onStateChange': onPlayerStateChange
    }
  });
}

function onPlayerReady(event) {
  event.target.playVideo();
  checkVideoTime();
}


function getCurrentTagGram(current_seconds) {

  var current_chunk = Math.floor((current_seconds * sample_rate) / audio_length);

  var current_taggram = null;
  if (taggram != null) {
    current_taggram = taggram[current_chunk];
  }
  return current_taggram;

}

function checkVideoTime() {
  var current_seconds = player.getCurrentTime();
  console.log(current_seconds);

  var current_taggram = getCurrentTagGram(current_seconds);

  if (current_taggram != null) {
    
    var sorted_tags = [];
    for(var i = 0; i < current_taggram.length; i++) {
      var tag_score = current_taggram[i];
      var tag_label = tags[i];
      sorted_tags.push({tag: tag_label, score: tag_score});
    }

    sorted_tags = sorted_tags.sort(function (a, b) {
        return b.score - a.score;
    });

    var top_n = 10;
    sorted_tags = sorted_tags.slice(0, top_n);

    var new_scores = sorted_tags.map(a => a.score);
    var new_tags = sorted_tags.map(a => a.tag);
    console.log(new_scores);

    if (tagChart != null) {
      tagChart.data.labels = new_tags;
      tagChart.data.datasets[0].data = new_scores;
      tagChart.update();
    }



    // var top_n = 5;
    // sorted_tags = sorted_tags.slice(0, top_n);

    // var tag_player_div = document.getElementById("tagPlayer");
    // tag_player_div.innerHTML = sorted_tags.map(m => {
    //   return `<div class="flex-column">${m.key}</div>`;
    // }).join('');

    console.log(sorted_tags);
  }


  setTimeout(checkVideoTime, 250);
}

// 5. The API calls this function when the player's state changes.
//    The function indicates that when playing a video (state=1),
//    the player should play for six seconds and then stop.
var done = false;
function onPlayerStateChange(event) {

  console.log(event);

  // if (event.data == YT.PlayerState.PLAYING && !done) {
  //   setTimeout(stopVideo, 6000);
  //   done = true;
  // }
}
function stopVideo() {
  player.stopVideo();
}



$("#player").hide();
$("#tagForm").submit(function (e) {

  e.preventDefault(); // avoid to execute the actual submit of the form.

  var form = $(this);
  var url = form.attr('action');

  $.ajax({
    type: "POST",
    url: url,
    data: form.serialize(), // serializes the form's elements.
    success: function (data) {

      // set image
      $("#taggramImage").attr("src", "/static/images/" + data["image"])

      var videoId = data["video_id"]
      player.loadVideoById(videoId)
      $("#player").show()

      taggram = data["taggram"]
      tags = data["tags"];


      var ctx = document.getElementById('tagChart').getContext('2d');
      tagChart = new Chart(ctx, {
        type: 'horizontalBar',
        data: {
          labels: tags,
          datasets: [{
            label: 'Tag Predictions',
            data: data["scores"],
            backgroundColor: [
              "#003f5c",
              "#2f4b7c",
              "#665191",
              "#a05195",
              "#d45087",
              "#f95d6a",
              "#ff7c43",
              "#ffa600"], 
            borderWidth: 1
          }]
        },
        options: {
          tooltips: {enabled: false},
          scales: {
            yAxes: [{
              ticks: {
                beginAtZero: true
              }
            }]
          }
        }
      });


      var ctx = document.getElementById('myChart').getContext('2d');
      var fullChart = new Chart(ctx, {
        type: 'horizontalBar',
        data: {
          labels: tags,
          datasets: [{
            label: 'Average Tag Predictions',
            data: data["scores"],
            borderWidth: 1
          }]
        },
        options: {
          scales: {
            yAxes: [{
              ticks: {
                beginAtZero: true
              }
            }]
          }
        }
      });
    }
  });


});

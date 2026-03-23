async function uploadVideo(){

    let fileInput = document.getElementById("videoInput")

    let file = fileInput.files[0]

    let formData = new FormData()

    formData.append("video",file)

    let response = await fetch(
        "http://127.0.0.1:5000/analyze_video",
        {
            method:"POST",
            body:formData
        }
    )

    let data = await response.json()

    drawTimeline(data.timeline)

    drawDistribution(data.timeline)
}


function drawTimeline(timeline){

    let times = timeline.map(t=>t.time)

    let emotionMap={
        angry:0,
        sad:1,
        neutral:2,
        happy:3
    }

    let values = timeline.map(t=>emotionMap[t.emotion])

    new Chart(document.getElementById("timelineChart"),{

        type:"line",

        data:{
            labels:times,

            datasets:[{
                label:"Emotion",

                data:values
            }]
        },

        options:{
            scales:{
                y:{
                    ticks:{
                        callback:function(value){

                            let map=["angry","sad","neutral","happy"]

                            return map[value]
                        }
                    }
                }
            }
        }

    })

}


function drawDistribution(timeline){

    let count={
        angry:0,
        happy:0,
        sad:0,
        neutral:0
    }

    timeline.forEach(t=>{
        count[t.emotion]++
    })

    new Chart(document.getElementById("distributionChart"),{

        type:"pie",

        data:{

            labels:Object.keys(count),

            datasets:[{

                data:Object.values(count)

            }]
        }

    })

}
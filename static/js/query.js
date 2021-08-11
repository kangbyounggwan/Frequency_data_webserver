jQuery(document).ready(function(){
if(loc[1] == "a=123"){
$(".notice").attr('class','notice ab');
$("#notice a").attr('class','ab');
}else if(loc[1] == "a=345"){
$("#free").removeClass("free");
$("#free").addClass("free menuon");
}
});
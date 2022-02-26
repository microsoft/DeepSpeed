---
layout: archive
permalink: /tags/
---

{% assign sorted_tags = (site.tags | sort:0) %}
<ul class="tag-box">
	{% for tag in sorted_tags %}
		{% assign t = tag | first %}
		{% assign posts = tag | last %}
		<li><a href="#{{ t | downcase }}">{{ t }} <span class="size">({{ posts.size }})</span></a></li>
	{% endfor %}
</ul>

{% for tag in sorted_tags %}
  {% assign t = tag | first %}
  {% assign posts = tag | last %}

<h4 id="{{ t | downcase }}">{{ t }}</h4>
<ul>
{% for post in posts %}
  {% if post.tags contains t %}
    <li>
       <span class="date">{{ post.date | date: '%d %b %y' }}</span>:  <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endif %}
{% endfor %}
</ul>
{% endfor %}

---
layout: archive
permalink: /posts/
---

{% if paginator %}
  {% assign posts = paginator.posts %}
{% else %}
  {% assign posts = site.posts %}
{% endif %}

<script type="text/javascript">
    function filterUsingCategory(selectedCategory) {
      var id = 0;
      {% for post in posts %}
        var cats = {{ post.tags | jsonify }}

        var postDiv = document.getElementById(++id);
        postDiv.style.display = (selectedCategory == 'All' || cats.includes(selectedCategory))
          ? 'unset'
          : 'none';
      {% endfor %}
    }
  </script>

  <div>
    <button id="All" onclick="filterUsingCategory('All')">*Show All Posts*</button>
    {% assign tags = site.tags | sort %}
    {% for category in tags %}
      {% assign cat = category | first %}
      <button id="{{ cat }}" onclick="filterUsingCategory(this.id)">{{ cat }}</button>
    {% endfor %}
    <hr />
  </div>

  <div class="posts-wrapper">
    {% assign id = 0 %}
    {% for post in posts %}
      {% assign id = id | plus:1 %}
      <div class="post" id="{{id}}">
        <p class="itemInteriorSection">
          <a href="{{post.url}}">{{ post.articletitle }}</a>
            {%- unless post.hidden -%}
              {% include archive-single.html %}
              {% if post.image %}
                <a href="{{ post.link }}"><img src="{{ post.image }}"></a>
              {% endif %}
            {%- endunless -%}
        </p>
      </div>
    {% endfor %}
  </div>


